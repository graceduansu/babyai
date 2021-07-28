import gym
from babyai.levels.levelgen import *

register_levels('iclr19_levels', globals())


env = gym.make('BabyAI-GoToImpUnlock-v0')

observation_space = env.observation_space
action_space = env.action_space

import copy
import gym
import time
import datetime
import numpy as np
import sys
import itertools
import torch
from babyai.evaluate import batch_evaluate
import babyai.utils as utils
from babyai.rl import DictList
from babyai.model import ACModel
import multiprocessing
import os
import json
import logging
import numpy 
import tqdm
import pickle

d_path = "/data/graceduansu/demos/BabyAI-GoToImpUnlock-v0"

demos_path = utils.get_demos_path(d_path, None, None, valid=False)
train_demos = utils.load_demos(demos_path)

rand_inds = np.random.choice(len(train_demos), size=6400)

some_demos = [train_demos[i] for i in rand_inds]
demo_list = utils.demos.transform_demos(some_demos)

from gym_minigrid.minigrid import *

for i in range(0, 128):
    print(i)
    batch = []
    concept_inds = []

    # filter for 1280 demos that have unlock action
    count = 0
    while count < 1280:
        unlock_idx = None
        pickup_idx = None

        while unlock_idx is None or pickup_idx is None:
            rand_idx = np.random.choice(len(some_demos))
            dem = some_demos[rand_idx]
            dem_transformed = demo_list[rand_idx]

            # if demo has Toggle (unlock) action
            if env.actions.toggle in dem[3] and env.actions.pickup in dem[3]:
                # color = None
                # for t in range(len(dem_transformed)):
                #     isdoor = (dem_transformed[t][0]['image'][3][5][0] == 4)
                #     istoggle = (dem[3][t] == env.actions.toggle)
                #     if isdoor and istoggle:
                #         unlock_idx = t
                #         color = dem_transformed[unlock_idx][0]['image'][3][5][1]
                    
                #     if color:
                #         iskey = (dem_transformed[t][0]['image'][3][5][0] == 5)
                #         iscolor = (dem_transformed[t][0]['image'][3][5][1] == color)
                #         if iskey and iscolor:
                #             pickup_idx = t
                #             break

                color = None
                for t in range(len(dem_transformed)):
                    iskey = (dem_transformed[t][0]['image'][3][5][0] == 5)
                    ispickup = (dem[3][t] == env.actions.pickup)
                    if iskey and ispickup:
                        pickup_idx = t
                        color = dem_transformed[pickup_idx][0]['image'][3][5][1]
                    
                    if color:
                        isdoor = (dem_transformed[t][0]['image'][3][5][0] == 4)
                        iscolor = (dem_transformed[t][0]['image'][3][5][1] == color)
                        istoggle = (dem[3][t] == env.actions.toggle)
                        if isdoor and iscolor and istoggle:
                            unlock_idx = t
                            break

        # (start, end) idx
        batch.append(dem_transformed)
        concept_inds.append((pickup_idx, unlock_idx))

        count += 1

    batch.sort(key=len, reverse=True)
    # Constructing flat batch and indices pointing to start of each demonstration
    cuda6 = torch.device('cuda:6')
    flat_batch = []
    inds = [0]
    flat_concept_inds = [(0,0)]

    for demo in batch:
        flat_batch += demo
        inds.append(inds[-1] + len(demo))
        flat_concept_inds.append((inds[-1] + pickup_idx, inds[-1] + unlock_idx))

    flat_batch = np.array(flat_batch)
    inds = inds[:-1]
    flat_concept_inds = flat_concept_inds[:-1]
    flat_concept_inds = flat_concept_inds[1:]
    num_frames = len(flat_batch)

    mask = np.ones([len(flat_batch)], dtype=np.float64)
    mask[inds] = 0
    mask = torch.tensor(mask, device=cuda6, dtype=torch.float).unsqueeze(1)

    # get batch with only concepts
    flat_concept_batch = []
    concept_inds = [0]
    for pair in flat_concept_inds:
        concept_data = flat_batch[pair[0]:pair[1]]
        flat_concept_batch.extend(concept_data)
        concept_inds.append(concept_inds[-1] + len(concept_data))

    print(concept_inds[-100:])
    concept_inds = concept_inds[:-3]
    print(concept_inds[-100:])
    concept_mask = np.ones([len(flat_concept_batch)], dtype=np.float64)
    concept_mask[concept_inds] = 0
    concept_mask = torch.tensor(concept_mask, device=cuda6, dtype=torch.float).unsqueeze(1)

    # Observations, true action, values and done for each of the stored demostration
    obss, action_true, done = flat_batch[:, 0], flat_batch[:, 1], flat_batch[:, 2]

    episode_ids = np.zeros(len(flat_batch))
    inds_copy = inds.copy()

    # Loop terminates when every observation in the flat_batch has been handled
    while True:
        # taking observations and done located at inds
        done_step = done[inds]
        episode_ids[inds] = range(len(inds))

        # Updating inds, by removing those indices corresponding to which the demonstrations have finished
        inds = inds[:len(inds) - sum(done_step)]
        if len(inds) == 0:
            break

        # Incrementing the remaining indices
        inds = [index + 1 for index in inds]

    # make concept_episode_ids
    flat_concept_batch = np.array(flat_concept_batch)
    obss, action_true, done = flat_concept_batch[:, 0], flat_concept_batch[:, 1], flat_concept_batch[:, 2]

    concept_episode_ids = np.zeros(len(flat_concept_batch))
    concept_inds_copy = concept_inds.copy()

    # Loop terminates when every observation in the flat_batch has been handled
    while True:
        # taking observations and done located at inds
        done_step = done[concept_inds]
        concept_episode_ids[concept_inds] = range(len(concept_inds))

        # Updating inds, by removing those indices corresponding to which the demonstrations have finished
        concept_inds = concept_inds[:len(concept_inds) - sum(done_step)]
        if len(concept_inds) == 0:
            break

        # Incrementing the remaining indices
        concept_inds = [index + 1 for index in concept_inds]

    # {flat_batch, mask, episode_ids, inds_copy, 
    #   flat_batch_concept_inds, 
    #   concept_mask, concept_episode_ids, concept_inds} <-- inds after batch only has concepts
    batch_dict = {'flat_batch': flat_batch,
                        'mask': mask,
                        'episode_ids': episode_ids,
                        'inds': inds_copy,
                        'flat_batch_concept_inds': flat_concept_inds,
                        'concept_mask': concept_mask,
                        'concept_episode_ids': concept_episode_ids,
                        'concept_inds': concept_inds_copy}

    path = '/data/graceduansu/concepts/1_search_for_key/batch' + str(i)
    with open(path, 'wb') as handle:
        pickle.dump(batch_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
