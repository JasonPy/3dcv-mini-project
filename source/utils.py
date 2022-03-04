import numpy as np
import time

def millis():
    return round(time.time() * 1000)

def is_valid_depth(depth_map: np.array, p: np.array, invalid_value=65535) -> bool:
    """
    Check if a pixel lookup is valid by checking for invalid depth value
    and if the pixel coordinate lays inside the image boundary
    """
    if depth_map[p[0],p[1]] == invalid_value or not in_bounds(depth_map.shape, p):
        return False
    else:
        return True

def in_bounds(matrix_shape: tuple, index: np.array) -> bool:
    """
    Check if a given index is in the range of a matrix boundary
    by determining if its value is below zero or above the image size
    """
    # try:
    if (index[0] < 0) or (index[1] < 0) or (index[0] > matrix_shape[0]) or (index[1] > matrix_shape[1]):
        return False
    else:
        return True