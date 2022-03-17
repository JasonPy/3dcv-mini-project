import pickle
import numpy as np
from sklearn import model_selection

from regression_forest import RegressionForest, objective_reduction_in_variance, param_sampler
from feature_extractor import FeatureType
from data_loader import DataLoader

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

DATA_PATH = '../data'
SCENE = 'fire'
TEST_SIZE = 0.5

NUM_TREES = 5
NUM_TRAIN_IMAGES_PER_TREE = 500
NUM_SAMPLES_PER_IMAGE = 5000
NUM_PARAMETER_SAMPLES = 1024


loader = DataLoader(DATA_PATH)
image_indices = np.arange(loader.get_dataset_length(SCENE))
train_indices, test_indices = model_selection.train_test_split(image_indices, test_size=TEST_SIZE)

forest = RegressionForest(
    num_trees = NUM_TREES,
    max_depth = 16,
    feature_type = FeatureType.DA_RGB,
    param_sampler = param_sampler,
    objective_function = objective_reduction_in_variance)

forest.train(
    data_dir = DATA_PATH,
    scene_name = SCENE,
    train_image_indices = train_indices,
    num_images_per_tree = NUM_TRAIN_IMAGES_PER_TREE,
    num_samples_per_image = NUM_SAMPLES_PER_IMAGE,
    num_param_samples = NUM_PARAMETER_SAMPLES)

if forest.is_trained:
    filename = 'trained_forest_1024.pkl'
    save_object(forest, filename)
    print(f'Done training. Saved as {filename}')
else:
    print(f'Training interrupted')