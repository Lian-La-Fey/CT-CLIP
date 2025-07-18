# from pathlib import Path
# from shutil import rmtree
# from transformer_maskgit.optimizer import get_optimizer
# from transformers import BertTokenizer, BertModel

# from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis

# from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score

# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader, random_split
# from torch.utils.data.distributed import DistributedSampler

# # from data_inference_nii import CTReportDatasetinfer
# #from data_external_valid import CTReportDatasetinfer

from data_inference import CTReportDatasetinfer

# import numpy as np
# import tqdm
# import pandas as pd

# from einops import rearrange
# import accelerate
# from accelerate import Accelerator
# from accelerate import DistributedDataParallelKwargs
# import math
# import torch.optim.lr_scheduler as lr_scheduler
# from ct_clip import CTCLIP

# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.parallel_loader as pl
# import torch_xla.utils.utils as xu

# # helpers

# def tensor_to_nifti(tensor, path, affine=np.eye(4)):
#     """
#     Save tensor as a NIfTI file.

#     Args:
#         tensor (torch.Tensor): The input tensor with shape (D, H, W) or (C, D, H, W).
#         path (str): The path to save the NIfTI file.
#         affine (np.ndarray, optional): The affine matrix for the NIfTI file. Defaults to np.eye(4).
#     """

#     tensor = tensor.cpu()

#     if tensor.dim() == 4:
#         # Assume single channel data if there are multiple channels
#         if tensor.size(0) != 1:
#             print("Warning: Saving only the first channel of the input tensor")
#         tensor = tensor.squeeze(0)
#     tensor=tensor.swapaxes(0,2)
#     numpy_data = tensor.detach().numpy().astype(np.float32)
#     nifti_img = nib.Nifti1Image(numpy_data, affine)
#     nib.save(nifti_img, path)

# def exists(val):
#     return val is not None

# def noop(*args, **kwargs):
#     pass

# def cycle(dl):
#     while True:
#         for data in dl:
#             yield data

# def yes_or_no(question):
#     answer = input(f'{question} (y/n) ')
#     return answer.lower() in ('yes', 'y')

# def accum_log(log, new_logs):
#     for key, new_value in new_logs.items():
#         old_value = log.get(key, 0.)
#         log[key] = old_value + new_value
#     return log

# def apply_softmax(array):
#     """
#     Applies softmax function to a torch array.

#     Args:
#         array (torch.Tensor): Input tensor array.

#     Returns:
#         torch.Tensor: Tensor array after applying softmax.
#     """
#     softmax = torch.nn.Softmax(dim=0)
#     softmax_array = softmax(array)
#     return softmax_array


# class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
#     def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_warmup=10000, gamma=1.0, last_epoch=-1):
#         self.T_0 = T_0
#         self.T_mult = T_mult
#         self.eta_max = eta_max
#         self.T_warmup = T_warmup
#         self.gamma = gamma
#         self.T_cur = 0
#         self.lr_min = 0
#         self.iteration = 0

#         super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         if self.iteration < self.T_warmup:
#             lr = self.eta_max * self.iteration / self.T_warmup
#         else:
#             self.T_cur = self.iteration - self.T_warmup
#             T_i = self.T_0
#             while self.T_cur >= T_i:
#                 self.T_cur -= T_i
#                 T_i *= self.T_mult
#                 self.lr_min = self.eta_max * (self.gamma ** self.T_cur)
#             lr = self.lr_min + 0.5 * (self.eta_max - self.lr_min) * \
#                  (1 + math.cos(math.pi * self.T_cur / T_i))

#         self.iteration += 1
#         return [lr for _ in self.optimizer.param_groups]

#     def step(self, epoch=None):
#         if epoch is None:
#             epoch = self.last_epoch + 1
#         self.last_epoch = epoch
#         self._update_lr()
#         self._update_T()

#     def _update_lr(self):
#         self.optimizer.param_groups[0]['lr'] = self.get_lr()[0]

#     def _update_T(self):
#         if self.T_cur == self.T_0:
#             self.T_cur = 0
#             self.lr_min = 0
#             self.iteration = 0
#             self.T_0 *= self.T_mult
#             self.eta_max *= self.gamma

# class CTClipInference(nn.Module):
#     def __init__(
#         self,
#         CTClip: CTCLIP,
#         *,
#         num_train_steps,
#         batch_size,
#         data_folder="external_valid",
#         reports_file="data_reports.xslx",
#         lr = 1e-4,
#         wd = 0.,
#         max_grad_norm = 0.5,
#         save_results_every = 100,
#         save_model_every = 2000,
#         results_folder = './results',
#         labels = "labels.csv",
#         accelerate_kwargs: dict = dict(),
#         device,
#     ):
#         super().__init__()
#         ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
#         self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
#         self.CTClip = CTClip
#         self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
#         self.results_folder = results_folder
#         self.register_buffer('steps', torch.Tensor([0]))

#         self.num_train_steps = num_train_steps
#         self.batch_size = batch_size

#         all_parameters = set(CTClip.parameters())

#         self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

#         self.max_grad_norm = max_grad_norm
#         self.lr=lr
#         # Load the pre-trained weights
#         self.ds = CTReportDatasetinfer(data_folder=data_folder, csv_file=reports_file,labels=labels)

#         # Split dataset into train and validation sets


#         self.dl = DataLoader(
#             self.ds,
#             num_workers=6,
#             batch_size=1,
#             shuffle = True,
#         )
#         # prepare with accelerator
#         self.dl_iter=cycle(self.dl)
#         self.device = self.accelerator.device
#         self.CTClip.to(self.device)
#         self.lr_scheduler = CosineAnnealingWarmUpRestarts(self.optim,
#                                                   T_0=4000000,    # Maximum number of iterations
#                                                   T_warmup=10000, # Number of warmup steps
#                                                   eta_max=lr)   # Maximum learning rate


#         (
#  			self.dl_iter,
#             self.CTClip,
#             self.optim,
#             self.lr_scheduler
#         ) = self.accelerator.prepare(
#             self.dl_iter,
#             self.CTClip,
#             self.optim,
#             self.lr_scheduler
#         )

#         self.save_model_every = save_model_every
#         self.save_results_every = save_results_every
#         self.result_folder_txt = self.results_folder
#         self.results_folder = Path(results_folder)

#         self.results_folder.mkdir(parents=True, exist_ok=True)



#     def save(self, path):
#         if not self.accelerator.is_local_main_process:
#             return

#         pkg = dict(
#             model=self.accelerator.get_state_dict(self.CTClip),
#             optim=self.optim.state_dict(),
#         )
#         torch.save(pkg, path)

#     def load(self, path):
#         path = Path(path)
#         assert path.exists()
#         pkg = torch.load(path)

#         CTClip = self.accelerator.unwrap_model(self.CTClip)
#         CTClip.load_state_dict(pkg['model'])

#         self.optim.load_state_dict(pkg['optim'])

#     def print(self, msg):
#         self.accelerator.print(msg)


#     @property
#     def is_main(self):
#         return self.accelerator.is_main_process

#     def train_step(self):
#         device = self.device

#         steps = int(self.steps.item())


#         # logs
#         logs = {}



#         if True:
#             with torch.no_grad():

#                 models_to_evaluate = ((self.CTClip, str(steps)),)

#                 for model, filename in models_to_evaluate:
#                     model.eval()
#                     predictedall=[]
#                     realall=[]
#                     logits = []

#                     text_latent_list = []
#                     image_latent_list = []
#                     accession_names=[]
#                     pathologies = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']
#                     for i in tqdm.tqdm(range(len(self.ds))):
#                         valid_data, text, onehotlabels, acc_name = next(self.dl_iter)

#                         plotdir = self.result_folder_txt
#                         Path(plotdir).mkdir(parents=True, exist_ok=True)

#                         predictedlabels=[]
#                         onehotlabels_append=[]

#                         for pathology in pathologies:
#                             text = [f"{pathology} is present.", f"{pathology} is not present."]
#                             text_tokens=self.tokenizer(
#                                             text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

#                             output = model(text_tokens, valid_data.cuda(),  device=device)

#                             output = apply_softmax(output)

#                             append_out=output.detach().cpu().numpy()
#                             predictedlabels.append(append_out[0])

#                         predictedall.append(predictedlabels)
#                         realall.append(onehotlabels.detach().cpu().numpy()[0])
#                         accession_names.append(acc_name[0])

#                     realall=np.array(realall)
#                     predictedall=np.array(predictedall)

#                     np.savez(f"{plotdir}labels_weights.npz", data=realall)
#                     np.savez(f"{plotdir}predicted_weights.npz", data=predictedall)
#                     with open(f"{plotdir}accessions.txt", "w") as file:
#                         for item in accession_names:
#                             file.write(item + "\n")


#                     dfs=evaluate_internal(predictedall,realall,pathologies, plotdir)

#                     writer = pd.ExcelWriter(f'{plotdir}aurocs.xlsx', engine='xlsxwriter')

#                     dfs.to_excel(writer, sheet_name='Sheet1', index=False)

#                     writer.close()
#         self.steps += 1
#         return logs




#     def infer(self, log_fn=noop):
#         device = next(self.CTClip.parameters()).device
#         device=torch.device('cuda')
#         while self.steps < self.num_train_steps:
#             logs = self.train_step()
#             log_fn(logs)

#         self.print('Inference complete')

import torch, numpy as np

torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
torch.serialization.add_safe_globals([np.ndarray])
torch.serialization.safe_globals([np.ndarray])
torch.serialization.add_safe_globals([np.dtype])

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tqdm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from einops import rearrange
import math
import torch.optim.lr_scheduler as lr_scheduler
from pathlib import Path
import nibabel as nib
import os

# TPU için gerekli kütüphaneler
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.runtime as xr

# Özel modüller
from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer
from ct_clip import CTCLIP

from data_inference import CTReportDatasetinfer

from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis


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
        num_train_steps,
        batch_size,
        data_folder="external_valid",
        reports_file="data_reports.xlsx",
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
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.steps = torch.tensor(0, device=self.device)
        
        # Veri setini yükle
        self.ds = CTReportDatasetinfer(data_folder=data_folder, csv_file=reports_file,labels=labels)
        
        # TPU için dağıtılmış sampler oluştur
        sampler = torch.utils.data.distributed.DistributedSampler(
            self.ds,
            num_replicas=xr.world_size(),
            rank=xr.global_ordinal(),
            shuffle=False
        )
        
        # Veri yükleyiciyi oluştur
        self.dl = DataLoader(
            self.ds,
            sampler=sampler,
            num_workers=4,
            batch_size=batch_size,
            pin_memory=False,
            drop_last=True,
        )
        
        # TPU için paralel yükleyici
        self.dl_iter = pl.MpDeviceLoader(self.dl, self.device)
        
        # Çıktı klasörünü oluştur
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
                    
                    # Model çıktısını al
                    output = self.CTClip(text_tokens, valid_data, device=self.device)
                    output = apply_softmax(output)
                    
                    # Pozitif sınıfın olasılığını sakla
                    predicted_labels.append(output[0].item())
                
                # Sonuçları topla
                predicted_all.append(predicted_labels)
                # real_all.append(onehotlabels.cpu().numpy()[0])
                real_all.append(onehotlabels.cpu().numpy()[0].tolist())
                accession_names.append(acc_name[0])
                
        
        print(f"{xr.global_ordinal()} - predicted_all: {predicted_all}")
        print(f"{xr.global_ordinal()} - real_all: {real_all}")
        
        # xm.rendezvous('after_loop')
        # Tüm TPU'ların sonuçlarını topla
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
        
        # Sadece master TPU sonuçları kaydetsin
        if xm.is_master_ordinal():
            
            # Sonuçları kaydet
            np.savez(self.results_folder / "labels_weights.npz", data=real_all)
            np.savez(self.results_folder / "predicted_weights.npz", data=predicted_all)
            
            # Accession numaralarını kaydet
            with open(self.results_folder / "accessions.txt", "w") as f:
                for name in accession_names:
                    f.write(f"{name}\n")
            
            # Metrikleri hesapla
            df_results = evaluate_internal(
                predicted_all, real_all, pathologies, self.results_folder
            )
            
            # Excel'e kaydet
            df_results.to_excel(self.results_folder / "aurocs.xlsx", index=False)
            
            # ROC eğrilerini çiz
            # plot_roc(real_all, predicted_all, pathologies, self.results_folder)
            
            # Bootstrap güven aralıkları
            # bootstrap_results = bootstrap(real_all, predicted_all, n_bootstraps=1000)
            # bootstrap_results.to_excel(self.results_folder / "bootstrap_results.xlsx")
            
            xm.master_print(f"Inference complete. Results saved to {self.results_folder}")
        
        # Tüm TPU'ları senkronize et
        xm.rendezvous('inference_done')
        