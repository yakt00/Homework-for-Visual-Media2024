import os
import argparse
import clip
import lavis
import numpy as np
import termcolor
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import utils
from evaluate import generate_cirr_test_submissions
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--data_path", type=str, required=True)

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    data_path = args.data_path
      
    clip_model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
    clip_model = clip_model.float().eval().requires_grad_(False).to(device)
    blip_model, blip_transform, _ = lavis.models.load_model_and_preprocess(name='blip2_t5', model_type="pretrain_flant5xxl", is_eval=True, device=device)

    index_features, index_names = utils.get_image_features(device, data_path, 'test1', 'classic', preprocess, blip_transform, clip_model, batch_size=32)
    predicted_features, gt_img_ids, reference_names, query_ids = utils.predict(device, data_path, 'test1', 'relative', preprocess, blip_transform['eval'], clip_model,blip_model)
    result_metrics = generate_cirr_test_submissions(device, predicted_features, reference_names, gt_img_ids, index_features, index_names, query_ids)     
        


