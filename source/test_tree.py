import numpy as np
import os
from sklearn import model_selection
from datetime import datetime

from regression_forest import RegressionForest, objective_reduction_in_variance, param_sampler
from feature_extractor import FeatureType
from data_loader import DataLoader
from utils import save_object

# BASE PARAMETERS
TIMESTAMP = datetime.now()
DATA_PATH = '../data'
OUTPUT_PATH = '../output'
SCENE = 'pumpkin'

# TRAINING PARAMETERS
TEST_SIZE = 1/3
NUM_TREES = 5
TREE_MAX_DEPTH = 16
NUM_TRAIN_IMAGES_PER_TREE = 500
NUM_SAMPLES_PER_IMAGE = 5000
NUM_PARAMETER_SAMPLES = 1024
FEATURE_TYPE = FeatureType.DA_RGB

loader = DataLoader(DATA_PATH)
image_indices = np.arange(loader.get_dataset_length(SCENE))
train_indices, test_indices = model_selection.train_test_split(image_indices, test_size=TEST_SIZE)

forest = RegressionForest(
    num_trees = NUM_TREES,
    max_depth = TREE_MAX_DEPTH,
    feature_type = FEATURE_TYPE,
    num_param_samples = NUM_PARAMETER_SAMPLES,
    param_sampler = param_sampler,
    objective_function = objective_reduction_in_variance)

forest.train(
    data_dir = DATA_PATH,
    scene_name = SCENE,
    train_image_indices = train_indices,
    num_images_per_tree = NUM_TRAIN_IMAGES_PER_TREE,
    num_samples_per_image = NUM_SAMPLES_PER_IMAGE)

# save used parameters
params = {
  "TIMESTAMP": TIMESTAMP.strftime("%d/%m/%Y %H:%M:%S"),
  "SCENE": SCENE,
  "TEST_SIZE": TEST_SIZE,
  "NUM_TREES": NUM_TREES,
  "TREE_MAX_DEPTH": TREE_MAX_DEPTH,
  "NUM_TRAIN_IMAGES_PER_TREE": NUM_TRAIN_IMAGES_PER_TREE,
  "NUM_SAMPLES_PER_IMAGE": NUM_SAMPLES_PER_IMAGE,
  "NUM_PARAMETER_SAMPLES": NUM_PARAMETER_SAMPLES,
  "FEATURE_TYPE": FEATURE_TYPE,
  "TRAIN_INDICES": train_indices,
  "TEST_INDICES": test_indices
}

if forest.is_trained:
    # create new directory and save forest / parameters
    target_dir = os.path.join(OUTPUT_PATH, TIMESTAMP.strftime(f'%d-%m-%Y_%H-%M_{SCENE}'), '')
    os.makedirs(target_dir)
    save_object(forest, os.path.join(target_dir, f'trained_forest_{SCENE}.pkl'))
    save_object(params, os.path.join(target_dir, f'params_{SCENE}.pkl'))
    print(f'Done training forest of scene {SCENE}.')
else:
    print(f'Training interrupted!')
