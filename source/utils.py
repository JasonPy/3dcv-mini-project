import numpy as np
import pickle
import time

from numba import njit, objmode
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
    return np.sum(np.sqrt(np.sum((arr - mean)**2, axis=1))) / size

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

@njit
def get_intrinsic_camera_matrix(focal_length=(585,585), principle_point=(320, 240)) -> np.array:
    """
    Create the intrinsic camera matrix using the focal length and principle points
    given at the official 7-scenes data set webpage.

    Returns
    ----------
    K: np.array
        Intrinsic camera matrix
    """
    fx, fy = focal_length
    cx, cy = principle_point
    K = np.diag(np.array([fx, fy, 1]))
    K[0, 2] = cx
    K[1 ,2] = cy
    return K.astype(np.float64)

@njit
def mult_along_axis(A, B, axis):
    if A.shape[axis] != B.size:
        raise ValueError("Length of 'A' along the given axis must be the same as B.size")

    shape = np.swapaxes(A, A.ndim-1, axis).shape
    B_brc = np.broadcast_to(B, shape)
    B_brc = np.swapaxes(B_brc, A.ndim-1, axis)
    return A * B_brc

def load_object(filename):
    """
    Load an pickle file in a certain destination.
    """
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

def save_object(obj, filename):
    """
    Save an object via pickle to a certain location.
    """
    with open(filename, 'wb') as outp: 
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
