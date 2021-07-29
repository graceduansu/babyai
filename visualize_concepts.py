import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import tqdm
import babyai.utils as utils
import gym
# from babyai.model import ACModel

from babyai.iterative_normalization import iterative_normalization_py

### Constants

CONCEPTS = [
    '1_search_for_key', '2_take_key_to_door', '3_search_for_target'
]

### Helper Functions

def load(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def starting_indexes(num_frames, recurrence=80):
    if num_frames % recurrence == 0:
        return np.arange(0, num_frames, recurrence)
    else:
        return np.arange(0, num_frames, recurrence)[:-1]

### Activations

def get_activations(acmodel, batch_dict, observation_space, action_space, model_name, 
    recurrence=80, num_concepts=3, device='cuda:6'):

    outputs = []
    def hook(module, input, output):
        X_hat = iterative_normalization_py.apply(
            input[0], 
            module.running_mean, module.running_wm, module.num_channels, 
            module.T, module.eps, module.momentum, module.training)
        size_X = X_hat.size()
        size_R = module.running_rot.size()
        X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

        X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
        X_hat = X_hat.view(*size_X)

        outputs.append(X_hat.cpu().numpy())

    acmodel.cw_layer.register_forward_hook(hook)

    flat_batch = batch_dict['flat_batch']
    mask = torch.tensor(batch_dict['mask'], device=device)
    episode_ids = batch_dict['episode_ids']
    inds = batch_dict['inds']
    flat_batch_concept_inds = batch_dict['flat_batch_concept_inds']
    concept_mask = torch.tensor(batch_dict['concept_mask'], device=device)
    concept_episode_ids = batch_dict['concept_episode_ids']
    concept_inds = batch_dict['concept_inds']

    obss_preprocessor = utils.ObssPreprocessor(model_name, observation_space, None)

    num_frames = len(flat_batch)

    obss, action_true, done = flat_batch[:, 0], flat_batch[:, 1], flat_batch[:, 2]
    
    len_batch = 1280
    # Memory to be stored
    memories = torch.zeros([len(flat_batch), acmodel.memory_size], device=device)
    memory = torch.zeros([len_batch, acmodel.memory_size], device=device)

    preprocessed_first_obs = obss_preprocessor(obss[inds], device=device)
    instr_embedding = acmodel._get_instr_embedding(preprocessed_first_obs.instr)

    # Loop terminates when every observation in the flat_batch has been handled
    while True:
        # taking observations and done located at inds
        obs = obss[inds]
        done_step = done[inds]
        preprocessed_obs = obss_preprocessor(obs, device=device)

        with torch.no_grad():
            # taking the memory till len(inds), as demos beyond that have already finished
            new_memory = acmodel(
                preprocessed_obs,
                memory[:len(inds), :], instr_embedding[:len(inds)])['memory']

        memories[inds, :] = memory[:len(inds), :]
        memory[:len(inds), :] = new_memory

        # Updating inds, by removing those indices corresponding to which the demonstrations have finished
        inds = inds[:len(inds) - sum(done_step)]
        if len(inds) == 0:
            break

        # Incrementing the remaining indices
        inds = [index + 1 for index in inds]

    ############ CUT MEMORIES AND OBSS TO CONCEPT TIMESTEPS ONLY ####################
    flat_concept_batch = []

    for pair in flat_batch_concept_inds:
        concept_data = flat_batch[pair[0]:pair[1]]
        flat_concept_batch.extend(concept_data)

    concept_memories = torch.zeros([len(flat_concept_batch), acmodel.memory_size], device=device)
    mem_start = 0
    for pair in flat_batch_concept_inds:
        mem = memories[pair[0]:pair[1]]
        concept_memories[mem_start:mem_start + len(mem)] = mem
        mem_start += len(mem)

    # Here, actual backprop upto recurrence happens

    flat_concept_batch = np.array(flat_concept_batch)
    # total_frames = len(indexes) * recurrence
    obss, action_true, done = flat_concept_batch[:, 0], flat_concept_batch[:, 1], flat_concept_batch[:, 2]

    preprocessed_first_obs = obss_preprocessor(obss[concept_inds], device=device)
    instr_embedding = acmodel._get_instr_embedding(preprocessed_first_obs.instr)
    c_indexes = starting_indexes(len(flat_concept_batch))
    concept_memory = concept_memories[c_indexes]

    with torch.no_grad():
        for _ in tqdm.trange(recurrence):
            obs = obss[c_indexes]
            preprocessed_obs = obss_preprocessor(obs, device=device)
            mask_step = concept_mask[c_indexes]

            model_results = acmodel(
                preprocessed_obs, concept_memory * mask_step,
                instr_embedding[concept_episode_ids[c_indexes]])

            concept_memory = model_results['memory']

            c_indexes += 1

    activations = np.vstack(outputs).max((2, 3))[:, :num_concepts]
    return activations



##########################################
model_name = 'BabyAI-CW_best'
device = 'cuda:0'
#############################################

model_path = '/data/graceduansu/models/'+ model_name + '/model.pt'

concept_directory='/data/graceduansu/concepts'
concept_dirs = sorted([os.path.join(concept_directory, filename) for filename in os.listdir(concept_directory)])

# Models

acmodel = torch.load(model_path).to(device)

# Env
env_name = 'BabyAI-GoToImpUnlock-v0'
env = gym.make(env_name)
observation_space = env.observation_space
action_space = env.action_space

x = []
y = []

for concept_index, concept_dir in enumerate(tqdm.tqdm(concept_dirs, leave=False)):
    concept_paths = [os.path.join(concept_dir, f) for f in os.listdir(concept_dir)]
    activations = []

    for _ in tqdm.trange(1):
        concept_path = np.random.choice(concept_paths)
        
        batch_dict = None
        with open(concept_path, 'rb') as file:
            batch_dict = pickle.load(file)

        a = get_activations(acmodel, batch_dict, observation_space, action_space, model_name, device=device)
        activations.append(a)
        x.append(a)
        y.append(concept_index)


    z = np.vstack(activations).mean(axis=0)
    print('Concept {}'.format(concept_index+1))
    print(z)

    x_pos = [i for i, _ in enumerate(CONCEPTS)]
    plt.bar(x_pos, z)
    plt.xticks(x_pos, CONCEPTS)
    plt.title('Model {}:\n Mean activations for Concept {} Data Input'.format(model_name, concept_index+1))
    plt.savefig('/data/graceduansu/models/{}/mean_activations_concept_{}.png'.format(model_name, concept_index+1))
    plt.show()
    plt.clf()

x = np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import StandardScaler

# p = np.random.permutation(len(x))
# N = int(len(p) / 5)
# X_train, y_train = x[p[:N]], y[p[:N]]
# X_test, y_test = x[p[N:]], y[p[N:]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Decision Tree classifer object
clf = LogisticRegression()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("LR Accuracy:", metrics.accuracy_score(y_test, y_pred))

dt = DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = dt.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("DT Accuracy:", metrics.accuracy_score(y_test, y_pred))