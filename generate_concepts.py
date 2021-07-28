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


def get_concept_data_from_batch(batch, batch_concept_inds, concept_path):
    cuda6 = torch.device('cuda:6')

    flat_batch = []
    inds = [0]
    flat_batch_concept_inds = [(0,0)]

    for j in tqdm.tqdm(range(len(batch))):
        demo = batch[j]
        flat_batch += demo
        flat_batch_concept_inds.append((inds[-1] + batch_concept_inds[j][0], 
                                        inds[-1] + batch_concept_inds[j][1]))

        inds.append(inds[-1] + len(demo))
        

    flat_batch = np.array(flat_batch)
    inds = inds[:-1]
    flat_batch_concept_inds = flat_batch_concept_inds[:-1]
    flat_batch_concept_inds = flat_batch_concept_inds[1:]

    mask = np.ones([len(flat_batch)], dtype=np.float64)
    mask[inds] = 0
    mask = torch.tensor(mask, device=cuda6, dtype=torch.float).unsqueeze(1)

    # get batch with only concepts
    flat_concept_batch = []
    concept_inds = [0]
    for pair in flat_batch_concept_inds:
        concept_data = flat_batch[pair[0]:pair[1]]
        flat_concept_batch.extend(concept_data)
        concept_inds.append(concept_inds[-1] + len(concept_data))

    print(concept_inds[-10:])
    concept_inds = [concept_inds[i] for i in range(len(concept_inds)) if concept_inds[i] < len(flat_concept_batch)]
    print('--------------------------------------------------------------------')
    print(concept_inds[-10:])
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
                        'batch_concept_inds': batch_concept_inds,
                        'flat_batch_concept_inds': flat_batch_concept_inds,
                        'concept_mask': concept_mask,
                        'concept_episode_ids': concept_episode_ids,
                        'concept_inds': concept_inds_copy}

    path = concept_path + str(i)
    with open(path, 'wb') as handle:
        pickle.dump(batch_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_concept3_data_from_batch(batch, batch_concept_inds):
    cuda6 = torch.device('cuda:6')

    flat_batch = []
    inds = [0]
    flat_batch_concept_inds = [(0,0)]

    for j in tqdm.tqdm(range(len(batch))):
        demo = batch[j]
        flat_batch += demo
        flat_batch_concept_inds.append((inds[-1] + batch_concept_inds[j][0], 
                                        inds[-1] + batch_concept_inds[j][1]))

        flat_batch_concept_inds.append((inds[-1] + batch_concept_inds[j][2], 
                                        inds[-1] + batch_concept_inds[j][3]))

        inds.append(inds[-1] + len(demo))

    flat_batch = np.array(flat_batch)
    inds = inds[:-1]
    flat_batch_concept_inds = flat_batch_concept_inds[:-1]
    flat_batch_concept_inds = flat_batch_concept_inds[1:]

    mask = np.ones([len(flat_batch)], dtype=np.float64)
    mask[inds] = 0
    mask = torch.tensor(mask, device=cuda6, dtype=torch.float).unsqueeze(1)

    # get batch with only concepts
    flat_concept_batch = []
    concept_inds = [0]
    for pair in flat_batch_concept_inds:
        concept_data = flat_batch[pair[0]:pair[1]]
        flat_concept_batch.extend(concept_data)
        concept_inds.append(concept_inds[-1] + len(concept_data))

    print(concept_inds[-10:])
    concept_inds = [concept_inds[i] for i in range(len(concept_inds)) if concept_inds[i] < len(flat_concept_batch)]
    print('--------------------------------------------------------------------')
    print(concept_inds[-10:])
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
                        'batch_concept_inds': batch_concept_inds,
                        'flat_batch_concept_inds': flat_batch_concept_inds,
                        'concept_mask': concept_mask,
                        'concept_episode_ids': concept_episode_ids,
                        'concept_inds': concept_inds_copy}

    path = '/data/graceduansu/concepts/3_search_for_target/batch' + str(i)
    with open(path, 'wb') as handle:
        pickle.dump(batch_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

################################ RUN ###################################################
for i in range(0, 128):
    print(i)
    batch = []
    concept1_inds = []
    concept2_inds = []
    concept3_inds = []

    # filter for 1280 demos that have unlock action
    count = 0
    see_door_idx = None
    unlock_idx = None
    pickup_idx = None
    dem_transformed = None

    while count < 1280:
        see_door_idx = None
        unlock_idx = None
        pickup_idx = None
        dem_transformed = None

        while unlock_idx is None or pickup_idx is None or see_door_idx is None:

            see_door_idx = None
            unlock_idx = None
            pickup_idx = None
            dem_transformed = None

            rand_idx = np.random.choice(len(some_demos))
            dem = some_demos[rand_idx]
            dem_transformed = demo_list[rand_idx]

            # if demo has Toggle (unlock) action
            if env.actions.toggle in dem[3] and env.actions.pickup in dem[3]:
                color = None
                for t in range(len(dem_transformed)-1, -1, -1):
                    isdoor = (dem_transformed[t][0]['image'][3][5][0] == 4)
                    istoggle = (dem[3][t] == env.actions.toggle)
                    if isdoor and istoggle and unlock_idx is None:
                        unlock_idx = t
                        color = dem_transformed[unlock_idx][0]['image'][3][5][1]
                    
                    if color:
                        iskey = (dem_transformed[t][0]['image'][3][5][0] == 5)
                        iscolor = (dem_transformed[t][0]['image'][3][5][1] == color)
                        ispickup = (dem[3][t] == env.actions.pickup)
                        if iskey and iscolor and ispickup and pickup_idx is None:
                            pickup_idx = t

                        if pickup_idx is not None:
                            if isdoor and iscolor:
                                see_door_idx = t
                                break

        # (start, end) idx
        batch.append(dem_transformed)
        last_idx = len(dem_transformed)
        concept1_inds.append((see_door_idx, pickup_idx))
        concept2_inds.append((pickup_idx, unlock_idx))
        concept3_inds.append((0, see_door_idx, unlock_idx, last_idx))

        count += 1

    batch.sort(key=len, reverse=True)
    get_concept_data_from_batch(batch, concept1_inds, '/data/graceduansu/concepts/1_search_for_key/batch')
    get_concept_data_from_batch(batch, concept2_inds, '/data/graceduansu/concepts/2_take_key_to_door/batch')
    get_concept3_data_from_batch(batch, concept3_inds)

