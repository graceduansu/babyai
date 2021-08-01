import gym
from babyai.levels.levelgen import *

register_levels('bonus_levels', globals())


env = gym.make('BabyAI-UnlockRGB-v0')

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
import babyai.bot

d_path = "/data/graceduansu/demos/BabyAI-UnlockRGB-v0"

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
    mask = np.expand_dims(mask, axis=1)

    # get batch with only concepts
    flat_concept_batch = []
    temp_inds = [0]
    for pair in flat_batch_concept_inds:
        concept_data = flat_batch[pair[0]:pair[1]]
        flat_concept_batch.extend(concept_data)
        temp_inds.append(temp_inds[-1] + len(concept_data))

    #print(temp_inds[-10:])
    concept_inds = [temp_inds[i] for i in range(len(temp_inds)) if temp_inds[i] < len(flat_concept_batch)]
    #print('--------------------------------------------------------------------')
    #print(concept_inds[-10:])
    concept_mask = np.ones([len(flat_concept_batch)], dtype=np.float64)
    concept_mask[concept_inds] = 0
    concept_mask = np.expand_dims(concept_mask, axis=1)

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
    concept_inds_copy = [temp_inds[i] for i in range(len(temp_inds)) if temp_inds[i] < len(flat_concept_batch)]

    # Loop terminates when every observation in the flat_batch has been handled
    while True:
        # taking observations and done located at inds
        concept_inds = [concept_inds[i] for i in range(len(concept_inds)) if concept_inds[i] < len(flat_concept_batch)]
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


################################ RUN ###################################################
COLOR = 0

for i in range(0, 256):
    print(i)
    batch = []
    concept0_inds = []
    concept1_inds = []
    concept2_inds = []
    concept3_inds = []
    concept4_inds = []
    concept5_inds = []
    concept6_inds = []

    # filter for 1280 demos that have unlock action
    count = 0
    search_begin_idx = None
    unlock_idx = None
    pickup_idx = None
    dem_transformed = None

    while count < 256:
        search_begin_idx = None
        unlock_idx = None
        pickup_idx = None
        dem_transformed = None

        while unlock_idx is None or pickup_idx is None or search_begin_idx is None:

            search_begin_idx = None
            unlock_idx = None
            pickup_idx = None
            dem_transformed = None

            rand_idx = np.random.choice(len(some_demos))
            dem = some_demos[rand_idx]
            dem_transformed = demo_list[rand_idx]

            # if demo has Toggle (unlock) action
            if env.actions.toggle in dem[3] and env.actions.pickup in dem[3]:
                color = None
                for t in range(0, len(dem_transformed), 1):
                    subgoal = dem[4][t]
                    if isinstance(subgoal, babyai.bot.GoNextToSubgoal):
                        obj_type = getattr(subgoal.datum, 'type', None)
                        if obj_type == 'key' and search_begin_idx is None:
                            search_begin_idx = t
                            
                    iskey = (dem_transformed[t][0]['image'][3][5][0] == 5)
                    ispickup = (dem[3][t] == env.actions.pickup)
                    if iskey and ispickup and pickup_idx is None:
                        pickup_idx = t
                        color = dem_transformed[pickup_idx][0]['image'][3][5][1]
                        if color != COLOR:
                            break

                    isdoor = (dem_transformed[t][0]['image'][3][5][0] == 4)
                    istoggle = (dem[3][t] == env.actions.toggle)
                    if isdoor and istoggle and unlock_idx is None:
                        unlock_idx = t
                        break
                    

        # (start, end) idx
        batch.append(dem_transformed)
        last_idx = len(dem_transformed)
        if COLOR == 0:
            # Red
            concept0_inds.append((search_begin_idx, pickup_idx))
            concept3_inds.append((pickup_idx, unlock_idx))
        elif COLOR == 1:
            # Green
            concept1_inds.append((search_begin_idx, pickup_idx))
            concept4_inds.append((pickup_idx, unlock_idx))
        elif COLOR == 2:
            # Blue
            concept2_inds.append((search_begin_idx, pickup_idx))
            concept5_inds.append((pickup_idx, unlock_idx))

        count += 1

    batch.sort(key=len, reverse=True)
    get_concept_data_from_batch(batch, concept0_inds, '/data/graceduansu/UnlockRGB_concepts/0_search_for_red_key/batch')
    #get_concept_data_from_batch(batch, concept1_inds, '/data/graceduansu/UnlockRGB_concepts/1_search_for_green_key/batch')
    #get_concept_data_from_batch(batch, concept2_inds, '/data/graceduansu/UnlockRGB_concepts/2_search_for_blue_key/batch')
    get_concept_data_from_batch(batch, concept3_inds, '/data/graceduansu/UnlockRGB_concepts/3_take_red_key_to_door/batch')
    #get_concept_data_from_batch(batch, concept4_inds, '/data/graceduansu/UnlockRGB_concepts/4_take_green_key_to_door/batch')
    #get_concept_data_from_batch(batch, concept5_inds, '/data/graceduansu/UnlockRGB_concepts/5_take_blue_key_to_door/batch')
