def in_bounds(self, matrix_shape: tuple, index: int) -> bool:
    """
    Check if a given index is in the range of a matrix boundary
    """
    for s in matrix_shape[:2]:
        if index[0] > s or index[0] < 0:
            return False
    return True

def is_valid_index(self, matrix: np.array, index: int) -> bool:
    """
    Check if a pixel lookup is valid by checking for None value
    and if the pixel coordinate lays inside the image boundary
    """
    if not in_bounds(matrix.shape, index) or matrix[index[0], index[1]] is None:
        return False
    else: 
        return True