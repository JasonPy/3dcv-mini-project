

import pickle
import numpy as np

from data_loader import DataLoader
import utils
import ransac
import time
def load_object(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

SCENE = 'chess'
DATA_PATH = '/home/marven/Programs/Studium/CV-Project/3dcv-mini-project/data'
NUM_TEST_IMAGES = 2
PREFIX = '/home/marven/Programs/Studium/CV-Project/3dcv-mini-project/output/23-03-2022_23-19_chess/'
loader = DataLoader(DATA_PATH)

params = load_object(f'{PREFIX}params_{SCENE}.pkl')

forest = load_object(f'{PREFIX}trained_forest_{SCENE}.pkl')

test_set_indices = params['TEST_INDICES']
test_indices = np.random.choice(test_set_indices, NUM_TEST_IMAGES, replace = False)
# images_data = loader.load_dataset(SCENE, [91,837])


loader = DataLoader(DATA_PATH)
images_data = loader.load_dataset('chess', [2830])

print(images_data[0][0].shape)
for index in range(images_data[0][0].shape[0]):    
    rgb_image = images_data[0][index]

    depth_image = images_data[1][0]
    depth_image = depth_image[:, :, np.newaxis]
    image = np.concatenate((rgb_image, depth_image), axis = 2)

    inv_camera_matrix = np.linalg.inv(utils.get_intrinsic_camera_matrix())
    t1 = time.time()
    hypa, energy = ransac.optimize(forest, image, 500, index, images_data, inv_camera_matrix)
    time_estimated = time.time() - t1
    print(time_estimated)
    np.savetxt(f"energy_scene_{index}", np.array([energy]))
    np.savetxt(f"camera_pose_scene_{index}", hypa)