import numpy as np
from utils import is_valid_index, in_bounds

class FeatureExtracor:
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

    def depth_feature(self, p: np.array, params: any) -> bool:
        """
        For an pixel coordinate p and corresponding image depth map
        obtain the image feature according to variant 'Depth'

        Parameters
        ----------
        p: np.array 
            containing the pixel coordinates (x,y)
        params: 
            parameters defined for each node

        Returns
        -------
        feature value for pixel at position p
        """
        (tau, delta1, delta2, c1, c2, z) = params
        
        # check if the depth value is defined for initial pixel coordinate p
        if is_valid_index(self.depth_data, p):
            p1 = p + delta1/self.depth_data[p[0], p[1]] 
            p2 = p + delta2/self.depth_data[p[0], p[1]] 
        else:
            p1 = p + delta1/self.depth_fill_value 
            p2 = p + delta2/self.depth_fill_value

        # make sure to use integer coordinates
        p1 = int(p1)
        p2 = int(p2)

        # check if new pixel lookup with p1 and p2 is valid
        d1 = self.depth_data[p1[0],p1[1]] if is_valid_index(self.depth_data, p1) else self.depth_fill_value
        d2 = self.depth_data[p2[0],p2[1]] if is_valid_index(self.depth_data, p2) else self.depth_fill_value

        return d1 - d2 >= tau
  
    def da_rgb_feature(self, p: np.array, params: any) -> bool:
        """
        For an pixel coordinate p, corresponding image and depth map
        obtain the image feature according to variant 'Depth-Adaptive RGB'

        Parameters
        ----------
        p: np.array 
            containing the pixel coordinates (x,y)
        params: 
            parameters defined for each node

        Returns
        ----------
        feature value for pixel at position p
        """
        (tau, delta1, delta2, c1, c2, z) = params

        # check if the depth value is defined for initial pixel coordinate p
        if is_valid_index(self.depth_data, p):
            p1 = p + delta1/self.depth_data[p[0], p[1]] 
            p2 = p + delta2/self.depth_data[p[0], p[1]] 
        else:
            p1 = p + delta1/self.depth_fill_value 
            p2 = p + delta2/self.depth_fill_value

        # make sure to use integer coordinates
        p1 = int(p1)
        p2 = int(p2)

        # TODO: what is the fill value here? neglect this pixel?
        i1 = self.rgb_data[p1[0], p1[1], c1] if in_bounds(self.depth_data.shape, p1) else None
        i2 = self.rgb_data[p2[0], p2[1], c2] if in_bounds(self.depth_data.shape, p2) else None

        return i1 - i2 >= tau

    def da_rgb_d_feature(self, p: np.array, params: any) -> bool:
        # TODO: how to combine depth and da_rgb feature values?
        pass