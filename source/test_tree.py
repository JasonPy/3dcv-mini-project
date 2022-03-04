from regression_forest import RegressionForest, objective_reduction_in_variance
from feature_extractor import FeatureType
from data_holder import sample_from_data_set
import data_loader

loader = data_loader.DataLoader('../data')
dataset = loader.load_dataset('heads', (0, 5))

import numpy as np
from numpy.random import choice, uniform

def param_sampler(num_samples: int) -> np.array:
    rgb_coords = np.array([0, 1, 2])
    tau = np.tile([0], num_samples)
    delta1 = uniform(-130, 130, num_samples)
    delta2 = uniform(-130, 130, num_samples)
    c1 = choice(rgb_coords, num_samples, replace=True)
    c2 = choice(rgb_coords, num_samples, replace=True)
    return np.array([tau, delta1, delta2, c1, c2]).transpose()

(data_rgb, data_d, data_pose) = dataset

data = sample_from_data_set(
    images_rgb=data_rgb,
    images_depth=data_d,
    camera_poses=data_pose,
    num_samples=500)

forest = RegressionForest(
    num_trees=1,
    max_depth=16,
    feature_type=FeatureType.DA_RGB,
    param_sampler=param_sampler,
    objective_function=objective_reduction_in_variance)


import cProfile
cProfile.run('forest.train(data, num_param_samples=128)')
