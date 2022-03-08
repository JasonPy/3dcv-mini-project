import pickle

from data_loader import DataLoader
from regression_forest import RegressionForest, objective_reduction_in_variance, param_sampler
from feature_extractor import FeatureType

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

loader = DataLoader('../data')
images_data = loader.load_dataset('heads', (0, 100))

data = loader.sample_from_data_set(
    images_data = images_data,
    num_samples = 5000)

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
    reset = False)

if forest.is_trained:
    filename = 'trained_forest_1024.pkl'
    save_object(forest, filename)
    print(f'Done training. Saved as {filename}')
else:
    print(f'Training interrupted')
