from enum import Enum
from typing import Tuple
import numpy as np

class FeatureType(Enum):
    """
    This bass class represents the three different feature types
    including 'Depth', 'Depth-Adaptive RGB' and 'Depth-Adaptive RGB + Depth'
    """
    DEPTH = 1
    DA_RGB = 2
    DA_RGB_DEPTH = 3


class FeatureExtractor:
    """
    Class for holding references to the image data needed for the feature
    extraction functions.
    """

    def __init__(self, image_data: Tuple[np.array, np.array, np.array], depth_fill_value: float = 6000):
        """
        Construct a feature extractor for the given image

        Parameters
        ----------
        index: int
        the image index
        image_data: Tuple[np.array, np.array, np.array]
        the images data
        depth_fill_value: float
        fill invalid depth values with distance in millimeters

        """
        self.rgb_data = image_data[0]
        self.depth_data = image_data[1]
        self.pose_data = image_data[2]
        self.depth_fill_value = depth_fill_value

    def get_features_for_samples(self, idx_s: np.array, p_s: np.array, param_sample: any, feature_type: FeatureType):
        """
        For a given pixel coordinate (sample), evaluate the feature function
        in the context of this feature space (image) according to the given
        parameters

        Parameters
        ----------
        idx_s: np.array
            containing the samples image ids
        p_s: np.array 
            containing the samples pixel coordinates (x,y)
        params: any
            parameters defined for each node,
        feature_type: FeatureType
            the feature-type to evaluate

        Returns
        ----------
        bool feature for this coordinate
        """
        (tau, delta1x, delta1y, delta2x, delta2y, c1, c2) = param_sample

        samples = np.vstack((idx_s, p_s.T))
        depths = self.depth_data[tuple(samples)]
        m, n = self.depth_data.shape[1:]

        depths_zero_mask = depths == 0
        depths[depths_zero_mask] = 6000

        p_delta1x = (samples[1] + delta1x // depths).astype(np.int16)
        p_delta1y = (samples[2] + delta1y // depths).astype(np.int16)
        p_delta2x = (samples[1] + delta2x // depths).astype(np.int16)
        p_delta2y = (samples[2] + delta2y // depths).astype(np.int16)


        outside_bounds_x = lambda x_s: np.logical_or(x_s < 0, x_s >= m)
        outside_bounds_y = lambda y_s: np.logical_or(y_s < 0, y_s >= n)
        mask_valid = np.any([
            depths_zero_mask,
            outside_bounds_x(p_delta1x),
            outside_bounds_x(p_delta2x),
            outside_bounds_y(p_delta1y),
            outside_bounds_y(p_delta2y)
            ], axis=0) == False

        num_valid = sum(mask_valid)
        s_delta1 = np.stack((samples[0], p_delta1x, p_delta1y))[:,mask_valid]
        s_delta2 = np.stack((samples[0], p_delta2x, p_delta2y))[:,mask_valid]

        if feature_type is FeatureType.DEPTH:
            depths_delta1 = self.depth_data[tuple(s_delta1)]
            depths_delta2 = self.depth_data[tuple(s_delta2)]
            mask_split = (depths_delta1.astype(int) - depths_delta2) >= tau
            return mask_valid, mask_split
        
        if feature_type is FeatureType.DA_RGB:
            c_delta1 = np.vstack((s_delta1, np.full(num_valid, c1, dtype=np.int16)))
            c_delta2 = np.vstack((s_delta2, np.full(num_valid, c2, dtype=np.int16)))
            rgb_delta1 = self.rgb_data[tuple(c_delta1)]
            rgb_delta2 = self.rgb_data[tuple(c_delta2)]
            mask_split = rgb_delta1.astype(int) - rgb_delta2 >= tau
            return mask_valid, mask_split

        elif feature_type is FeatureType.DA_RGB_DEPTH:
            raise NotImplementedError()

    def generate_data_samples(self, index: int, num_samples: int) -> np.array:
        """
        Draw random coordinates from the image and calculate corresponding
        3d points in image coordinates and word coordinates (using camera pose)      
        """
        m, n = self.depth_data.shape[1:]
        coordinate_range = np.arange(n * m, dtype=np.uint16)
        indices = np.random.choice(coordinate_range, num_samples, replace=True)
        p = np.array([indices % m, indices // m], dtype=np.uint16).T
        p_depth = np.array([(x, y, self.depth_data[index, x, y], 1) for (x, y) in p], dtype=np.uint16)
        
        m = np.array([(self.pose_data[index] @ p)[:-1] for p in p_depth])
        return p, m