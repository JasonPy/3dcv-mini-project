from enum import Enum
from typing import Union
import numpy as np
from utils import is_valid_depth, in_bounds

class FeatureType(Enum):
    """
    This bass class represents the three different feature types
    including 'Depth', 'Depth-Adaptive RGB' and 'Depth-Adaptive RGB + Depth'
    """
    DEPTH = 1
    DA_RGB = 2
    DA_RGB_DEPTH = 3


class FeatureExtractor():
    """
    Class for holding references to the image data needed for the feature
    extraction functions.
    """

    def __init__(self, depth_data: np.array, rgb_data: np.array, camera_pose: np.array, depth_fill_value: float = 6000):
        """
        Construct a feature extractor for the given image

        Parameters
        ----------
        depth_data: np.array
            the images depth values in millimeters
        rgb_data: np.array
            the image as rgb 
        camera_pose: np.array
            the image's camera pose expressed as a 4x4 transformation matrix
        depth_fill_value: float
            fill invalid depth values with distance in millimeters

        """
        self.depth_data = depth_data
        self.rgb_data = rgb_data
        self.camera_pose = camera_pose
        self.depth_fill_value = depth_fill_value

    def get_feature(self, p: np.array, params: any, type: FeatureType) -> Union[bool, None]:
        """
        For a given pixel coordinate (sample), evaluate the feature function
        in the context of this feature space (image) according to the given
        parameters

        Parameters
        ----------
        p: np.array 
            containing the pixel coordinates (x,y)
        params: 
            parameters defined for each node

        Returns
        ----------
        bool feature for this coordinate
        """

        (tau, delta1, delta2, c1, c2, z) = params

        # check if the depth value is defined for initial pixel coordinate p
        if is_valid_depth(self.depth_data, p):
            p1 = p + delta1/self.depth_data[p[1], p[0]] 
            p2 = p + delta2/self.depth_data[p[1], p[0]] 
        else:
            return None

        # make sure to use integer coordinates
        p1 = p1.astype(int)
        p2 = p2.astype(int)

        if type is FeatureType.DEPTH:
            if is_valid_depth(self.depth_data, p1) and is_valid_depth(self.depth_data, p2):
                d1 = self.depth_data[p1[1],p1[0]] 
                d2 = self.depth_data[p2[1],p2[0]]
                return d1 - d2 >= tau
            else:
                return None

        elif type is FeatureType.DA_RGB:
            if in_bounds(self.depth_data.shape, p1) and in_bounds(self.depth_data.shape, p2):
                i1 = self.rgb_data[p1[1], p1[0], c1] 
                i2 = self.rgb_data[p2[1], p2[0], c2]
                return i1 - i2 >= tau
            else:
                return None

        elif type is FeatureType.DA_RGB_DEPTH:
            raise NotImplementedError()

    def generate_data_samples(self, num_samples: int) -> np.array:
        """
        Draw random coordinates from the image and calculate corresponding
        3d points in image coordinates and word coordinates (using camera pose)      
        """
        m, n = self.depth_data.shape

        indices = np.random.choice(m * n, num_samples, replace=True)
        p = np.array([(c % m, c // m) for c in indices])
        p_depth = np.array([(x, y, self.depth_data[x, y], 1) for (x, y) in p])

        m = (self.camera_pose @ p_depth)[:,:-1]
        # TODO: make sure this actually works
        return np.array(p, m)