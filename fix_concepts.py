import torch
import pickle
import tqdm
import os
import numpy as np

concept_directory='/data/graceduansu/concepts'
concept_dirs = sorted([os.path.join(concept_directory, filename) for filename in os.listdir(concept_directory)])

for concept_index, concept_dir in enumerate(tqdm.tqdm(concept_dirs, leave=False)):
    concept_paths = [os.path.join(concept_dir, f) for f in os.listdir(concept_dir)]
    for concept_path in tqdm.tqdm(concept_paths):
        batch_dict = None
        with open(concept_path, 'rb') as file:
            batch_dict = pickle.load(file)

        flat_batch = batch_dict['flat_batch']
        mask = batch_dict['mask']
        episode_ids = batch_dict['episode_ids']
        inds = batch_dict['inds']
        batch_concept_inds = batch_dict['batch_concept_inds']
        flat_batch_concept_inds = batch_dict['flat_batch_concept_inds']
        concept_mask = batch_dict['concept_mask']
        concept_episode_ids = batch_dict['concept_episode_ids']
        concept_inds = batch_dict['concept_inds']

        mask = mask.cpu().numpy()
        concept_mask = concept_mask.cpu().numpy()

        batch_dict = {'flat_batch': flat_batch,
                        'mask': mask,
                        'episode_ids': episode_ids,
                        'inds': inds,
                        'batch_concept_inds': batch_concept_inds,
                        'flat_batch_concept_inds': flat_batch_concept_inds,
                        'concept_mask': concept_mask,
                        'concept_episode_ids': concept_episode_ids,
                        'concept_inds': concept_inds}

        with open(concept_path, 'wb') as handle:
            pickle.dump(batch_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
