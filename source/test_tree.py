import pickle

import numpy as np
from numpy.random import choice, uniform

from data_holder import ProcessingPool, sample_from_data_set
from data_loader import DataLoader

from regression_forest import RegressionForest, objective_reduction_in_variance
from feature_extractor import FeatureType

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

loader = DataLoader('../data')
images_data = loader.load_dataset('heads', (0, 500))

data = sample_from_data_set(
    images_data = images_data,
    num_samples = 5000)

processing_pool = ProcessingPool(images_data = images_data)

def param_sampler(num_samples: int) -> np.array:
    rgb_coords = np.array([0, 1, 2])
    tau = np.tile([0], num_samples)
    delta1 = uniform(-130, 130, num_samples)
    delta2 = uniform(-130, 130, num_samples)
    c1 = choice(rgb_coords, num_samples, replace=True)
    c2 = choice(rgb_coords, num_samples, replace=True)
    return np.array([tau, delta1, delta2, c1, c2]).transpose()

forest = RegressionForest(
    num_trees = 1,
    max_depth = 16,
    feature_type = FeatureType.DA_RGB,
    param_sampler = param_sampler,
    objective_function = objective_reduction_in_variance)

print(f'Training forest with {len(data)} samples')

forest.train(
    data = data,
    processing_pool = processing_pool,
    num_param_samples = 128,
    reset = False)

filename = 'trained_forest.pkl'
save_object(forest, filename)
processing_pool.stop_workers()

print(f'Done training. Saved as {filename}')