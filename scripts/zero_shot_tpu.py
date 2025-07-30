import torch, numpy as np

torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
torch.serialization.add_safe_globals([np.ndarray])
torch.serialization.safe_globals([np.ndarray])
torch.serialization.add_safe_globals([np.dtype])

import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

import math
import torch.optim.lr_scheduler as lr_scheduler
from pathlib import Path
import nibabel as nib
import os

# TPU
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.runtime as xr

from transformers import BertTokenizer
from ct_clip import CTCLIP

from eval import evaluate_internal

from data_inference_nii import CTReportDatasetinfer

# Yardımcı fonksiyonlar
def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def tensor_to_nifti(tensor, path, affine=np.eye(4)):
    tensor = tensor.cpu()
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.swapaxes(0, 2)
    numpy_data = tensor.detach().numpy().astype(np.float32)
    nifti_img = nib.Nifti1Image(numpy_data, affine)
    nib.save(nifti_img, path)

def apply_softmax(array):
    softmax = torch.nn.Softmax(dim=0)
    return softmax(array)

# Özel learning rate scheduler
class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_warmup=10000, gamma=1.0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_warmup = T_warmup
        self.gamma = gamma
        self.T_cur = 0
        self.lr_min = 0
        self.iteration = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.iteration < self.T_warmup:
            lr = self.eta_max * self.iteration / self.T_warmup
        else:
            self.T_cur = self.iteration - self.T_warmup
            T_i = self.T_0
            while self.T_cur >= T_i:
                self.T_cur -= T_i
                T_i *= self.T_mult
                self.lr_min = self.eta_max * (self.gamma ** self.T_cur)
            lr = self.lr_min + 0.5 * (self.eta_max - self.lr_min) * \
                 (1 + math.cos(math.pi * self.T_cur / T_i))
        self.iteration += 1
        return [lr for _ in self.optimizer.param_groups]

# Ana inference sınıfı
class CTClipInference(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        *,
        batch_size,
        data_folder="external_valid",
        reports_file="data_reports.xlsx",
        meta_file="meta_data.csv",
        labels="labels.csv",
        results_folder='./results',
        device=None,
    ):
        super().__init__()
        self.device = device or xm.xla_device()
        self.CTClip = CTClip.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(
            'microsoft/BiomedVLP-CXR-BERT-specialized',
            do_lower_case=True
        )
        self.results_folder = results_folder
        self.batch_size = batch_size
        self.steps = torch.tensor(0, device=self.device)
        
        self.ds = CTReportDatasetinfer(data_folder=data_folder, reports_file=reports_file, meta_file=meta_file, labels=labels)
        
        # create distributed sampler for TPU
        sampler = torch.utils.data.distributed.DistributedSampler(
            self.ds,
            num_replicas=xr.world_size(),
            rank=xr.global_ordinal(),
            shuffle=False
        )
        
        self.dl = DataLoader(
            self.ds,
            sampler=sampler,
            num_workers=4,
            batch_size=batch_size,
            pin_memory=False,
            drop_last=True,
        )
        
        # parallel loader for TPU
        self.dl_iter = pl.MpDeviceLoader(self.dl, self.device)
        
        # create the output folder
        if xm.is_master_ordinal():
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(parents=True, exist_ok=True)
        
    def save(self, path):
        if xm.is_master_ordinal():
            pkg = {
                'model': self.CTClip.state_dict(),
                'optim': self.optim.state_dict() if hasattr(self, 'optim') else None,
            }
            torch.save(pkg, path)

    def load(self, path):
        if xm.is_master_ordinal():
            state_dict = torch.load(path)
            self.CTClip.load_state_dict(state_dict['model'])
        xm.rendezvous('load_model')

    def infer(self):
        xm.master_print(f"Starting inference on {xr.world_size()} TPU cores")
        
        # Patolojilerin tanımlanması
        pathologies = [
            'Medical material', 'Arterial wall calcification', 'Cardiomegaly',
            'Pericardial effusion', 'Coronary artery wall calcification', 'Hiatal hernia',
            'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule',
            'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
            'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation',
            'Bronchiectasis', 'Interlobular septal thickening'
        ]
        
        # Sonuçları saklamak için listeler
        predicted_all = []
        real_all = []
        accession_names = []
        
        total_batches = math.ceil(len(self.ds) / (self.batch_size * xr.world_size()))
        
        # Inference döngüsü
        self.CTClip.eval()
        with torch.no_grad():
            for i, (valid_data, text, onehotlabels, acc_name) in enumerate(self.dl_iter, start=1):
                # Veriyi TPU'ya taşı
                valid_data = valid_data.to(self.device)
                onehotlabels = onehotlabels.to(self.device)
                print(f"[TPU {xr.global_ordinal()}] Batch {i}/{total_batches}")
                
                # Her patoloji için sıfırdan öğrenme
                predicted_labels = []
                for pathology in pathologies:
                    # Pozitif ve negatif metin promptları
                    text_prompts = [
                        f"{pathology} is present.",
                        f"{pathology} is not present."
                    ]
                    
                    # Metni tokenize et ve TPU'ya taşı
                    text_tokens = self.tokenizer(
                        text_prompts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=512
                    ).to(self.device)
                    
                    # model çıktısını al
                    output = self.CTClip(text_tokens, valid_data, device=self.device)
                    output = apply_softmax(output)
                    
                    # Pozitif sınıfın olasılığını sakla
                    predicted_labels.append(output[0].item())
                
                # Collect results
                predicted_all.append(predicted_labels)
                # real_all.append(onehotlabels.cpu().numpy()[0])
                real_all.append(onehotlabels.cpu().numpy()[0].tolist())
                accession_names.append(acc_name[0])
                
        
        print(f"{xr.global_ordinal()} - predicted_all: {predicted_all}")
        print(f"{xr.global_ordinal()} - real_all: {real_all}")
        
        # xm.rendezvous('after_loop')
        # collect results of all TPUs
        predicted_all = xm.mesh_reduce(
            'pred',
            predicted_all,
            lambda replicas: np.vstack([
                np.array(sample, dtype=np.float32)
                for replica_list in replicas
                for sample       in replica_list
            ])
        )
        real_all = xm.mesh_reduce(
            'real',
            real_all,
            lambda replicas: np.vstack([
                np.array(sample, dtype=np.int32)
                for replica_list in replicas
                for sample       in replica_list
            ])
        )
        accession_names = xm.mesh_reduce('acc', accession_names, lambda x: sum(x, []))
        
        print(f"{xr.global_ordinal()} - accession_names")
        
        # only master TPU saves results
        if xm.is_master_ordinal():
            np.savez(self.results_folder / "labels_weights.npz", data=real_all)
            np.savez(self.results_folder / "predicted_weights.npz", data=predicted_all)
            
            # save accession numbers
            with open(self.results_folder / "accessions.txt", "w") as f:
                for name in accession_names:
                    f.write(f"{name}\n")
            
            # calc metrics
            df_results = evaluate_internal(
                predicted_all, real_all, pathologies, self.results_folder
            )
            
            # excel'e kaydet
            df_results.to_excel(self.results_folder / "aurocs.xlsx", index=False)
            
            xm.master_print(f"Inference complete. Results saved to {self.results_folder}")
        
        # Synchronize TPUs
        xm.rendezvous('inference_done')
        