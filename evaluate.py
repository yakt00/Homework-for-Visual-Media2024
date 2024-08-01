import json
import os

import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import tqdm


@torch.no_grad()
def generate_cirr_test_submissions(device, predicted_features, reference_names, targets, index_features, index_names, query_ids):
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)
    
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    reference_mask = torch.tensor(sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)
    
    targets = np.array(targets)
    group_mask = (sorted_index_names[..., None] == targets[:, None, :]).sum(-1).astype(bool)

    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)
    pairid_to_retrieved_images, pairid_to_group_retrieved_images = {}, {}
    for pair_id, prediction in zip(query_ids, sorted_index_names):
        pairid_to_retrieved_images[str(int(pair_id))] = prediction[:50].tolist()
    for pair_id, prediction in zip(query_ids, sorted_group_names):
        pairid_to_group_retrieved_images[str(int(pair_id))] = prediction[:3].tolist()            

    submission = {'version': 'rc2', 'metric': 'recall'}
    group_submission = {'version': 'rc2', 'metric': 'recall_subset'}

    submission.update(pairid_to_retrieved_images)
    group_submission.update(pairid_to_group_retrieved_images)

    submissions_folder_path = os.path.join(os.getcwd(), 'data', 'test_submissions', 'cirr')
    os.makedirs(submissions_folder_path, exist_ok=True)

    with open(os.path.join(submissions_folder_path, preload_dict['test']), 'w') as file:
        json.dump(submission, file, sort_keys=True)
    with open(os.path.join(submissions_folder_path, f"subset_{preload_dict['test']}"), 'w') as file:
        json.dump(group_submission, file, sort_keys=True)                        
    return None

