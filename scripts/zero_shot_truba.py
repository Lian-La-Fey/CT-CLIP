import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertTokenizer
from eval import evaluate_internal
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from data_inference_nii import CTReportDatasetinfer
from ct_clip import CTCLIP


# helpers
def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def apply_softmax(array):
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
    softmax = torch.nn.Softmax(dim=0)
    softmax_array = softmax(array)
    return softmax_array


class CTClipInference(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        *,
        data_folder="external_valid",
        reports_file="data_reports.xslx",
        meta_file="meta_data.csv",
        results_folder = './results',
        labels = "labels.csv",
        batch_size=1,  # Add batch_size parameter
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        self.results_folder = results_folder
        self.batch_size = batch_size  # Store batch size
        self.register_buffer('steps', torch.Tensor([0]))
        
        self.ds = CTReportDatasetinfer(
            data_folder=data_folder, 
            reports_file=reports_file, 
            meta_file=meta_file, 
            labels=labels
        )
        
        self.sampler = DistributedSampler(
            self.ds,
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            shuffle=False
        )
        
        self.dl = DataLoader(
            self.ds,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=8,
            shuffle=False,  # sampler handles shuffling
            pin_memory=True
        )

        self.device = self.accelerator.device
        self.CTClip.to(self.device)
        
        (
            self.dl,
            self.CTClip,
        ) = self.accelerator.prepare(
            self.dl,
            self.CTClip,
        )

        self.result_folder_txt = self.results_folder
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def infer(self, log_fn=noop):
        device = self.device
        steps = int(self.steps.item())
        logs = {}
        
        with torch.no_grad():
            model = self.CTClip
            model.eval()
            pathologies = [
                'Medical material','Arterial wall calcification', 'Cardiomegaly', 
                'Pericardial effusion','Coronary artery wall calcification', 
                'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 
                'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 
                'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 
                'Consolidation', 'Bronchiectasis','Interlobular septal thickening'
            ]
            
            # initialize storage for results
            all_predicted = []
            all_real = []
            all_accessions = []
            
            # Use accelerator's progress bar
            progress_bar = tqdm(
                range(len(self.dl)), 
                disable=not self.accelerator.is_main_process
            )
            
            for batch in self.dl:
                valid_data, text, onehotlabels, acc_name = batch
                batch_size = valid_data.shape[0]
                
                # Process batch
                predictedlabels = []
                for pathology in pathologies:
                    text_prompts = [
                        f"{pathology} is present.",
                        f"{pathology} is not present."
                    ]
                    text_tokens = self.tokenizer(
                        text_prompts, 
                        return_tensors="pt", 
                        padding="max_length", 
                        truncation=True, 
                        max_length=512
                    ).to(device)
                    
                    # Expand image data to match text prompts
                    expanded_images = valid_data.repeat_interleave(2, dim=0)
                    
                    # Get model output
                    output = model(text_tokens, expanded_images, device=device)
                    output = output.view(batch_size, 2)
                    output = apply_softmax(output)
                    
                    # Get probability for "present"
                    present_prob = output[:, 0].detach().cpu().numpy()
                    predictedlabels.append(present_prob)
                
                # Store results for this batch
                for i in range(batch_size):
                    sample_predicted = [pathology_probs[i] for pathology_probs in predictedlabels]
                    all_predicted.append(sample_predicted)
                    all_real.append(onehotlabels[i].detach().cpu().numpy())
                    all_accessions.append(acc_name[i])
                
                progress_bar.update(1)
            
            # Gather results from all processes
            all_predicted = self.accelerator.gather(all_predicted)
            all_real = self.accelerator.gather(all_real)
            all_accessions = self.accelerator.gather(all_accessions)
            
            # Only main process saves results
            if self.accelerator.is_main_process:
                plotdir = self.result_folder_txt
                Path(plotdir).mkdir(parents=True, exist_ok=True)
                
                # Convert to numpy arrays
                all_predicted = np.array(all_predicted)
                all_real = np.array(all_real)
                
                # Save results
                np.savez(f"{plotdir}labels_weights.npz", data=all_real)
                np.savez(f"{plotdir}predicted_weights.npz", data=all_predicted)
                
                with open(f"{plotdir}accessions.txt", "w") as file:
                    for item in all_accessions:
                        file.write(item + "\n")
                
                # Evaluate and save AUROCs
                dfs = evaluate_internal(all_predicted, all_real, pathologies, plotdir)
                writer = pd.ExcelWriter(f'{plotdir}aurocs.xlsx', engine='xlsxwriter')
                dfs.to_excel(writer, sheet_name='Sheet1', index=False)
                writer.close()
        
        self.steps += 1
        log_fn(logs)
        self.print('Inference complete')