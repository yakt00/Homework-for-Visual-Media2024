import json
import os
import time
import sys
import argparse
import clip
import numpy as np
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 
import pickle
import torch
import tqdm
import prompts
from loader_cirr import construct_loader

if torch.cuda.is_available():
    dtype = torch.float16
else:
    dtype = torch.float32


openai.api_key = "<api_key>"
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(60))
def openai_completion(prompt, engine="gpt-3.5-turbo", max_tokens=700, temperature=0, api_key=None):
    if api_key is not None:
        openai.api_key = api_key
    resp =  openai.ChatCompletion.create(
        model=engine,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        timeout=10,
        request_timeout=10,
        temperature=temperature,
        stop=["\n\n", "<|endoftext|>"],
        )
    
    return resp['choices'][0]['message']['content']


@torch.no_grad()
def get_image_features(device, data_path, split, mode, transform, blip_transform, model, batch_size):
    index_features, index_names = [], []
    for img_info in tqdm.tqdm(construct_loader(data_path, split, mode, transform, blip_transform, batch_size)):
        if mode == 'relative':
            images = img_info.get('reference_image')
            names = img_info.get('reference_name')
        elif mode == 'classic':
            images = img_info.get('image')
            names = img_info.get('image_name')
        with torch.no_grad():
            features = model.encode_image(images.to(device))
            index_features.append(features.cpu())
            index_names += names
    index_features = torch.vstack(index_features)
        
    return index_features, index_names
    
def get_text_features(device, model, captions, batch_size=32):
    features = []
    for i in tqdm.trange(int(np.ceil(len(captions)/batch_size)), position=0):
        tmp_captions = captions[i*batch_size:(i+1)*batch_size]
        if hasattr(model, 'tokenizer'):
            tokenized_captions = model.tokenizer(tmp_captions, context_length=77).to(device)
        else:
            tokenized_captions = clip.tokenize(tmp_captions, context_length=77, truncate=True).to(device)
        text_features = model.encode_text(tokenized_captions)
        features.append(text_features)
    features = torch.nn.functional.normalize(torch.vstack(features), dim=-1)        
        
    return features

@torch.no_grad()
def predict(device, data_path, split, mode, preprocess, blip_transform, clip_model, blip_model):
    torch.cuda.empty_cache()    
    batch_size = 32

    query_loader = construct_loader(data_path, split, mode, preprocess, blip_transform, batch_size)            
    query_iterator = tqdm.tqdm(query_loader, position=0)
    
    reference_names = []
    relative_captions = []
    gt_ids = []
    query_ids = []
    captions = []
    for qry in query_iterator:
        blip_image = qry['blip_ref_img'].to(device)
        reference_names += qry['reference_name']
        relative_captions += qry['relative_caption']
        gt_ids += np.array(qry['group_members']).T.tolist()
        query_ids += qry['pair_id']

        query_iterator.set_postfix_str(f'Shape: {blip_image.size()}')
            
        tmp_captions = []
        blip_prompt = prompts.blip_prompt
        for i in tqdm.trange(blip_image.size(0), position=1, leave=False):
            img = blip_image[i].unsqueeze(0)
            caption = blip_model.generate({'image': img, "prompt": blip_prompt})
            tmp_captions.append(caption[0])
        captions += tmp_captions

    modified_captions = []
    base_prompt = prompts.simple_modifier_prompt
    for i in tqdm.trange(len(captions), position=1, leave=False):
        prompt = base_prompt + '\n' + "Image Content: " + captions[i] + '\n' + 'Instruction: ' + relative_captions[i]
        llm_output = openai_completion(prompt, api_key=openai.api_key)
        modified = 0
        for line in llm_output.split('\n'):
            if line.strip().startswith('Edited Description:'):
                modified_captions.append(line.split(':')[1].strip())
                modified = 1
        if modified == 0:
            modified_captions.append(relative_captions[i])
    
    predicted_features = get_text_features(device, clip_model, modified_captions, batch_size=batch_size)   
    
    return predicted_features, gt_img_ids, reference_names, query_ids

    
