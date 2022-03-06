from enum import Enum
from typing import Union, Tuple
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

    def __init__(self, index: int, image_data: Tuple[np.array, np.array, np.array], depth_fill_value: float = 6000):
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
        self.index = index
        self.rgb_data = image_data[0]
        self.depth_data = image_data[1]
        self.camera_pose = image_data[2]
        self.depth_fill_value = depth_fill_value

    def get_feature(self, p: np.array, params: any, feature_type: FeatureType) -> Union[bool, None]:
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
        def depth (p: np.array) -> int:
            d = self.depth_data[p[0], p[1]]
            return 6000 if d == 0 else d
        
        def rgb (p, c) -> float:
            if not in_bounds(self.rgb_data.shape, p):
                return 0
            else:
                return self.rgb_data[p[0], p[1], int(c)]


        (tau, delta1, delta2, c1, c2) = params
        # check if the depth value is defined for initial pixel coordinate p
        if is_valid_depth(self.depth_data, p):
            p1 = p + delta1 / depth(p)
            p2 = p + delta2 / depth(p)
        else:
            return False

        # make sure to use integer coordinates
        p1 = p1.astype(int)
        p2 = p2.astype(int)

        if feature_type is FeatureType.DEPTH:
            if is_valid_depth(self.depth_data, p1) and is_valid_depth(self.depth_data, p2):
                d1 = depth(p1)
                d2 = depth(p2)
                return d1 - d2 >= tau
            else:
                return False

        elif feature_type is FeatureType.DA_RGB:
            if in_bounds(self.depth_data.shape, p1) and in_bounds(self.depth_data.shape, p2):
                i1 = rgb(p1, c1)
                i2 = rgb(p2, c2)
                return i1 - i2 >= tau
            else:
                return False

        elif feature_type is FeatureType.DA_RGB_DEPTH:
            raise NotImplementedError()

    def generate_data_samples(self, num_samples: int) -> np.array:
        """
        Draw random coordinates from the image and calculate corresponding
        3d points in image coordinates and word coordinates (using camera pose)      
        """
        m, n = self.depth_data.shape

        indices = np.random.choice(m * n, num_samples, replace=True)
        p = np.array([[c % m, c // m] for c in indices])
        p_depth = np.array([(x, y, self.depth_data[x, y], 1) for (x, y) in p])

        m = np.array([(self.camera_pose @ p)[:-1] for p in p_depth])
        return p, m