from enum import Enum
from typing import Tuple
import numpy as np
from numba import njit
from utils import array_for_indices_3d, array_for_indices_4d, get_intrinsic_camera_matrix, mult_along_axis

class FeatureType(Enum):
    """
    This bass class represents the three different feature types
    including 'Depth', 'Depth-Adaptive RGB' and 'Depth-Adaptive RGB + Depth'
    """
    DEPTH = 1
    DA_RGB = 2
    DA_RGB_DEPTH = 3

@njit
def get_features_for_samples(image_data: Tuple[np.array, np.array, np.array], p_s: np.array, param_sample: any, feature_type: FeatureType):
    """
    For a given pixel coordinate (sample), evaluate the feature function
    in the context of this feature space (image) according to the given
    parameters

    Parameters
    ----------
    p_s: np.array 
        containing the samples pixel coordinates (idx, x, y) (idx == image index)
    params: any
        parameters defined for each node,
    feature_type: FeatureType
        the feature-type to evaluate

    Returns
    ----------
    bool feature for this coordinate
    """
    (tau, delta1x, delta1y, delta2x, delta2y, c1, c2) = param_sample

    m, n = image_data[1].shape[1:]
    depths = array_for_indices_3d(image_data[1], p_s)
    depths_zero_mask = depths == 0
    depths[depths_zero_mask] = 6000

    p_delta1x = (p_s[:,1] + delta1x // depths).astype(np.int16)
    p_delta1y = (p_s[:,2] + delta1y // depths).astype(np.int16)
    p_delta2x = (p_s[:,1] + delta2x // depths).astype(np.int16)
    p_delta2y = (p_s[:,2] + delta2y // depths).astype(np.int16)

    outside_bounds_x = lambda x_s: np.logical_or(x_s < 0, x_s >= m)
    outside_bounds_y = lambda y_s: np.logical_or(y_s < 0, y_s >= n)
    mask_delta1_bounds = np.logical_or(outside_bounds_x(p_delta1x), outside_bounds_y(p_delta1y))
    mask_delta2_bounds = np.logical_or(outside_bounds_x(p_delta2x), outside_bounds_y(p_delta2y))
    mask_delta_bounds = np.logical_or(mask_delta1_bounds, mask_delta2_bounds)
    mask_valid = ~np.logical_or(depths_zero_mask, mask_delta_bounds)

    num_valid = sum(mask_valid)
    s_delta1 = np.stack((p_s[:,0], p_delta1x, p_delta1y))[:,mask_valid].T
    s_delta2 = np.stack((p_s[:,0], p_delta2x, p_delta2y))[:,mask_valid].T

    if feature_type is FeatureType.DEPTH:
        depths_delta1 = array_for_indices_3d(image_data[1], s_delta1).astype(np.int16)
        depths_delta2 = array_for_indices_3d(image_data[1], s_delta2).astype(np.int16)
        mask_split = (depths_delta1 - depths_delta2) >= tau
        return mask_valid, mask_split
    
    if feature_type is FeatureType.DA_RGB:
        c_delta1 = np.vstack((s_delta1.T, np.expand_dims(np.full(num_valid, c1, dtype=np.int16), axis=0))).T
        c_delta2 = np.vstack((s_delta2.T, np.expand_dims(np.full(num_valid, c2, dtype=np.int16), axis=0))).T
        rgb_delta1 = array_for_indices_4d(image_data[0], c_delta1).astype(np.int16)
        rgb_delta2 = array_for_indices_4d(image_data[0], c_delta2).astype(np.int16)
        mask_split = rgb_delta1 - rgb_delta2 >= tau
        return mask_valid, mask_split

    elif feature_type is FeatureType.DA_RGB_DEPTH:
        raise NotImplementedError()

@njit
def generate_data_samples(image_data: Tuple[np.array, np.array, np.array], index: int, num_samples: int) -> np.array:
    """
    Draw random coordinates from the image and calculate corresponding
    3d points in image coordinates and word coordinates (using camera pose)      
    """
    valid_samples = 0
    p_s_all = np.zeros((num_samples, 3), dtype=np.int16)
    w_s_all = np.zeros((num_samples, 3), dtype=np.float64)

    while valid_samples < num_samples:
        samples_to_draw = num_samples - valid_samples
        m, n = image_data[1].shape[1:]
        coordinate_range = np.arange(n * m, dtype=np.int32)
        x_y_s = np.random.choice(coordinate_range, samples_to_draw, replace=True)
        x_s = x_y_s // n
        y_s = x_y_s % n

        idx_s = np.full(samples_to_draw, index, dtype=np.int16)
        p_s = np.stack((idx_s, x_s, y_s)).T
        depths = array_for_indices_3d(image_data[1], p_s)
        valid_depths_mask = depths != 65535
        
        # Pad with 1 to work in homogeneous coordinates
        hom_camera_coordinates = (
            x_s * depths / 1000,
            y_s * depths / 1000,
            depths / 1000,
            np.full(samples_to_draw, 1, dtype=np.int16))
        p_hom_d_s = np.stack(hom_camera_coordinates).T.astype(np.float64)
        intrinsic_matrix_inv = np.linalg.inv(get_intrinsic_camera_matrix())
        pose_matrix = image_data[2][index]

        w_s = np.zeros((samples_to_draw, 3), dtype=np.float64)
        for i, p_hom_d in enumerate(p_hom_d_s):
            p_hom_d = np.ascontiguousarray(p_hom_d)
            w_s[i] = (pose_matrix @ (intrinsic_matrix_inv @ p_hom_d))[:-1].astype(np.float64)

        num_valid_samples = np.sum(valid_depths_mask)
        p_s_all[valid_samples:valid_samples+num_valid_samples] = p_s[valid_depths_mask]
        w_s_all[valid_samples:valid_samples+num_valid_samples] = w_s[valid_depths_mask]
        valid_samples += num_valid_samples

    return p_s_all, w_s_all