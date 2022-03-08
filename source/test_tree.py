import pickle

import numpy as np
from numpy.random import choice, uniform

from data_loader import DataLoader
from regression_forest import RegressionForest, objective_reduction_in_variance
from feature_extractor import FeatureType

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

loader = DataLoader('../data')
images_data = loader.load_dataset('heads', (0, 500))

data = loader.sample_from_data_set(
    images_data = images_data,
    num_samples = 5000)

def param_sampler(num_samples: int) -> np.array:
    rgb_coords = np.array([0, 1, 2])
    tau = np.tile([0], num_samples)
    delta1x = uniform(-130 * 100, 130 * 100, num_samples)
    delta1y = uniform(-130 * 100, 130 * 100, num_samples)
    delta2x = uniform(-130 * 100, 130 * 100, num_samples)
    delta2y = uniform(-130 * 100, 130 * 100, num_samples)
    c1 = choice(rgb_coords, num_samples, replace=True)
    c2 = choice(rgb_coords, num_samples, replace=True)
    return np.array([tau, delta1x, delta1y, delta2x, delta2y, c1, c2]).transpose()

forest = RegressionForest(
    num_trees = 1,
    max_depth = 16,
    feature_type = FeatureType.DA_RGB,
    param_sampler = param_sampler,
    objective_function = objective_reduction_in_variance)

forest.train(
    images_data = images_data,
    data = data,
    num_param_samples = 1024,
    reset = False,
    num_workers = 8)

filename = 'trained_forest.pkl'
save_object(forest, filename)

print(f'Done training. Saved as {filename}')
