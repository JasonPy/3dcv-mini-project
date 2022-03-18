import numpy as np
from numba import njit, jit, objmode
import time
from typing import Tuple
from sklearn.cluster import MeanShift

def get_arrays_size_MB (arrays: Tuple[np.array]):
    return sum([arr.nbytes for arr in arrays]) / 1024 / 1024

def millis():
    return round(time.time() * 1000)

@njit
def array_for_indices_3d(array: np.array, indices: np.array):
    outputs = np.zeros(indices.shape[0], dtype=array.dtype)
    assert len(array.shape) == indices.shape[1]
    for i, idx in enumerate(indices):
        outputs[i] = array[idx[0], idx[1], idx[2]]
    return outputs

@njit
def array_for_indices_4d(array: np.array, indices: np.array):
    outputs = np.zeros(indices.shape[0], dtype=array.dtype)
    assert len(array.shape) == indices.shape[1]
    for i, idx in enumerate(indices):
        outputs[i] = array[idx[0], idx[1], idx[2], idx[3]]
    return outputs

@njit
def split_set(set: np.array, mask: np.array):
    set_left = set[mask]
    set_right = set[~mask]
    return set_left, set_right

@njit
def vector_3d_array_mean(arr: np.array):
    mean_0 = np.mean(arr[:,0])
    mean_1 = np.mean(arr[:,1])
    mean_2 = np.mean(arr[:,2])
    return np.array([mean_0, mean_1, mean_2], dtype=arr.dtype)

@njit
def vector_3d_array_variance(arr: np.array):
    size = arr.shape[0]
    mean = vector_3d_array_mean(arr)
    return np.sum(np.abs(arr - mean)) / size

@njit("float32[:](float32[:, :])")
def get_mode(arr: np.array):
    """
    Apply mean shift clustering to obtain modes. 
    Then find most frequent mode.

    Parameters
    ----------
    arr: np.array
        Array of 3D coordinates to be clustered

    Returns
    ----------
    mode: float
        Most frequent coordinate center after clustering
    """
    with objmode(labels_='int64[:]', cluster_centers_='float32[:,:]'):
        modes = MeanShift(bandwidth=.1).fit(arr)
        labels_ = modes.labels_
        cluster_centers_ = modes.cluster_centers_
    mode = np.bincount(labels_).argmax()
    return cluster_centers_[mode]

def get_intrinsic_camera_matrix(focal_length=(585,585), principle_point=(320,240)) -> np.array:
    fx, fy = focal_length
    cx, cy = principle_point

    K = np.diag([fx,fy,1])
    K[0,2] = cx
    K[1,2] = cy
    return K

def image_2_camera_coordinate(coordinates: np.array, depths: np.array, K: np.array) -> np.array:
    # TODO: whats the shape of the pixels (maybe transpose coordinates_h)
    coordinates_h = np.pad(coordinates,[(0,0),(0,1)], mode='constant', constant_values=1) # homogenize
    return np.linalg.inv(K) @ coordinates_h.T * depths