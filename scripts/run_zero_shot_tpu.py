import os
import torch

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu

from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from zero_shot_tpu import CTClipInference


def _mp_fn(index, args):
    device = xm.xla_device()
    
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8
    )
    
    clip = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=294912,
        dim_text=768,
        dim_latent=512,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False
    )
    
    checkpoint = torch.load(
        "/home/raspuntinov/gcs/ct_rate/models/ct_clip/ct_clip/CT-CLIP_v2.pt",
        map_location=torch.device('cpu'),
        weights_only=False,
    )
    
    filtered_checkpoint = {
        k: v for k, v in checkpoint.items()
        if k != "text_transformer.embeddings.position_ids"
    }

    clip.load_state_dict(filtered_checkpoint, strict=False)
    xm.rendezvous("load_model")
    
    clip = clip.to(device)
    
    inference = CTClipInference(
        clip,
        data_folder="/home/raspuntinov/gcs/ct_rate/dataset/valid_fixed",
        reports_file="/home/raspuntinov/gcs/ct_rate/dataset/radiology_text_reports/validation_reports.csv",
        labels="/home/raspuntinov/gcs/ct_rate/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
        meta_file="/home/raspuntinov/gcs/ct_rate/dataset/metadata/validation_metadata.csv",
        batch_size=4,
        results_folder="inference_zeroshot/",
        device=device
    )
    
    inference.infer()

if __name__ == '__main__':
    torch_xla.launch(_mp_fn, args=(None,), debug_single_process=False)