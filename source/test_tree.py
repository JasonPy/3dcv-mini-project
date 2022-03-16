import pickle
import numpy as np

from sklearn import model_selection

from data_loader import DataLoader
from regression_forest import RegressionForest, objective_reduction_in_variance, param_sampler
from feature_extractor import FeatureType

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

SCENE = 'heads'
TEST_SIZE = 0.5

loader = DataLoader('../data')
dataset_len = loader.get_dataset_length(SCENE)

# split dataset into training and test masks/indices
train_mask, test_mask = model_selection.train_test_split(np.arange(dataset_len), test_size=TEST_SIZE)

# load training data and sample pixels from it
train_images = loader.load_dataset(SCENE, train_mask) # rgb, d, pose
train_data = loader.sample_from_data_set(
    images_data = train_images,
    num_samples = 5000)

# load test data and sample pixels from it
test_images = loader.load_dataset(SCENE, test_mask)
test_data = loader.sample_from_data_set(
    images_data = test_images,
    num_samples = 5000)

forest = RegressionForest(
    num_trees = 1,
    max_depth = 4,
    feature_type = FeatureType.DA_RGB,
    param_sampler = param_sampler,
    objective_function = objective_reduction_in_variance)

forest.train(
    images_data = train_images,
    data = train_data,
    num_param_samples = 1024,
    reset = False)

if forest.is_trained:
    filename = 'trained_forest_1024.pkl'
    save_object(forest, filename)
    print(f'Done training. Saved as {filename}')
else:
    print(f'Training interrupted')