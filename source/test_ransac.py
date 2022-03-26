import pickle
import numpy as np
import ransac
import time
import os

from data_loader import DataLoader
from evaluator import PoseEvaluator

def parallel(ransac: ransac.Ransac):
    return ransac.find_poses()

def load_object(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

NUM_TEST_IMAGES = 3

SCENE = 'pumpkin'
DATA_PATH = '../data'
OUTPUT = '../output'
PREFIX = '25-03-2022_17-41_pumpkin'

loader = DataLoader(DATA_PATH)

params = load_object(os.path.join(OUTPUT, PREFIX, f"params_{SCENE}.pkl"))
forest = load_object(os.path.join(OUTPUT, PREFIX, f"trained_forest_{SCENE}.pkl"))

test_set_indices = params['TEST_INDICES']
test_indices = np.random.choice(test_set_indices, NUM_TEST_IMAGES, replace = False)

images_data = loader.load_dataset(SCENE, test_indices)
ground_truth_poses = images_data[2][:NUM_TEST_IMAGES, :, :]

ransac = ransac.Ransac(images_data, forest, np.arange(NUM_TEST_IMAGES))

t1 = time.time()
predicted_poses = ransac.find_poses()
print(time.time() - t1)

translation_err = PoseEvaluator().get_translational_error(predicted_poses, ground_truth_poses)
angular_err  = PoseEvaluator().get_angular_error(predicted_poses, ground_truth_poses)
total_err = PoseEvaluator().evaluate(predicted_poses, ground_truth_poses)

print(translation_err)
print(angular_err)
