import itertools
import numpy as np
import os

def generate_data(grid_dims, n):
    features = generate_features(grid_dims, n)
    labels = generate_labels(features)
    return features, labels

def generate_features(grid_dims, n):
    a, b = grid_dims
    probs = np.random.rand(n)
    features = [
        np.random.choice([0., 1.], size=(10, 10), p=(probs[i], 1 - probs[i]))
        for i in range(n)
    ]
    return features

def generate_labels(features):
    labels = [
        next_world_state(i)
        for i in features    
    ]
    return labels

def next_world_state(world):
    next_state = np.zeros(world.shape)
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            next_state[i, j] = next_cell_state((i, j), world)
    return next_state

def next_cell_state(cell, world):
    i, j = cell
    neighborhood = get_neighborhood(cell, world)
    n_alive_neighbors = sum([world[i, j] for (i, j) in neighborhood])
    current_state = world[i, j]
    next_state = life_rule(current_state, n_alive_neighbors)
    return next_state

def get_neighborhood(cell, world):
    i, j = cell
    dim_1_surrounding = [i - 1, i, i + 1]
    dim_2_surrounding = [j - 1, j, j + 1]
    indices = [
        p 
        for p in itertools.product(dim_1_surrounding, dim_2_surrounding)
        if p != (i, j)
    ]
    indices = [
        idx
        for idx in indices 
        if ((idx[0] != -1) & 
            (idx[1] != -1) & 
            (idx[0] < world.shape[0]) & 
            (idx[1] < world.shape[1]))
    ]
    return indices

def life_rule(current_state, n_alive_neighbors):
    alive_condition = alive_condition = (
        ((current_state == 1) & (n_alive_neighbors in [2, 3])) |  # survival condition
        ((current_state == 0) & (n_alive_neighbors == 3))  # reproduction condition
    )
    if alive_condition:
        next_state = 1
    else:
        next_state = 0    
    return next_state


features_train, labels_train = generate_data((10, 10), 1000)
features_val, labels_val = generate_data((10, 10), 100)
features_test, labels_test = generate_data((10, 10), 100)

if not "data" in os.listdir():
    os.mkdir("data")


np.save(os.path.join("data", "features_train.npy"), features_train, allow_pickle=True)
np.save(os.path.join("data", "features_val.npy"), features_val, allow_pickle=True)
np.save(os.path.join("data", "features_test.npy"), features_test, allow_pickle=True)
np.save(os.path.join("data", "labels_train.npy"), labels_train, allow_pickle=True)
np.save(os.path.join("data", "labels_val.npy"), labels_val, allow_pickle=True)
np.save(os.path.join("data", "labels_test.npy"), labels_test, allow_pickle=True)