

import pickle
import numpy as np

from data_loader import DataLoader
import utils
import ransac
import time
from multiprocessing import Pool


def get_angular_error(poses: np.array, ground_truth: np.array) -> np.array:
    """
    Calculate the angle between two sets of poses.
    Parameters
    ----------
    poses: np.array
        the poses predicted by some algorithm as an array of 4x4 matrices
    ground_truth: np.array
        the true poses as an array of 4x4 matrices
    Returns
    -------
    angular_error: np.array
        angular error between corresponding poses and ground truth
    """

    # obtain the rotation matrices from the pose matrices
    R_pos = poses[:,:3,:3]
    R_gt = ground_truth[:,:3,:3]

    # compute the difference matrix, representing the difference rotation
    R_diff = np.matmul(R_pos, np.transpose(R_gt, axes=(0,2,1)))

    # get the angle theta of difference rotations
    theta = (np.trace(R_diff, axis1=1, axis2=2) - 1) / 2
    angular_error = np.rad2deg(np.arccos(np.clip(theta, -1, 1)))

    return angular_error


def get_translational_error(poses: np.array, ground_truth: np.array) -> np.array:
    """
    The translational error is defined as the distance between two points.
    Parameters
    ----------
    poses: np.array
        the poses predicted by some algorithm as array of 4x4 matrices
    ground_truth: np.array
        the true poses as array of 4x4 matrices
    Returns
    -------
    translational_error: np.array
        translational error between corresponding poses
    """

    # obtain translation vectors from poses
    T_pos = poses[:,:3,3]
    T_gt = ground_truth[:,:3,3]

    # calculate translational error
    translational_error = np.linalg.norm(T_pos- T_gt, axis=1)

    return translational_error


def parallel(ransac: ransac.Ransac):
    return ransac.find_poses()

def load_object(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

NUM_TEST_IMAGES = 20

SCENE = 'pumpkin'
DATA_PATH = '/home/marven/Programs/Studium/CV-Project/3dcv-mini-project/data'

PREFIX = '/home/marven/Programs/Studium/CV-Project/3dcv-mini-project/output/26-03-2022_11_pumpkin/'
loader = DataLoader(DATA_PATH)

params = load_object(f'{PREFIX}params_{SCENE}.pkl')

forest = load_object(f'{PREFIX}trained_forest_{SCENE}.pkl')

test_set_indices = params['TEST_INDICES']
test_indices = np.random.choice(test_set_indices, NUM_TEST_IMAGES, replace = False)
# images_data = loader.load_dataset(SCENE, [91,837])

print(test_indices)
loader = DataLoader(DATA_PATH)
images_data = loader.load_dataset(SCENE, test_indices)
pose = images_data[2]

t = images_data[2][:NUM_TEST_IMAGES, :, :]
rans = []

# for i in range(NUM_TEST_IMAGES):
#         rans.append(ransac.Ransac(images_data, forest, np.array([i])))
# with Pool(processes=12) as pool:
#    r = pool.map(parallel, rans)



ransac = ransac.Ransac(images_data, forest, np.arange(NUM_TEST_IMAGES))

t1 = time.time()
poses = ransac.find_poses()
print(time.time() - t1)
a = 1
translation_err = get_translational_error(poses, images_data[2][:NUM_TEST_IMAGES, :, :])
angular_err  = get_angular_error(poses, images_data[2][:NUM_TEST_IMAGES, :, :])

print("asd")