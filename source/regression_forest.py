from nis import match
import numpy as np
from enum import Enum
import tqdm



class Node:
    def __init__(self, evalFunction, initialParams = []):
        self.params = initialParams
        self.evalFunction = evalFunction
        self.leftChild = None
        self.rightChild = None
        self.response = None

    def is_leaf(self):
        return (self.leftChild is None) and (self.leftRight is None)

    def evaluate(self, p):
        output = self.evalFunction(self.params, p)
        if self.is_leaf():
            return self.response

        nextNode = self.rightChild if output else self.leftChild
        return nextNode.evaulate(p)

    def train(self, data):
        # TODO: we need the corresponding image and depth map here to calculate features:
        # To evaluate a regression tree at a 2D pixel location p in an image, we start at 
        # the root node and descend to a leaf by repeatedly evaluating the weak learner
        # smt like ->   data = [[p, m, img, depth_map], [...]]
        data = [[p, m], [...]]

        param_cadidates = []
        # sample

        best_score_Q = None
        best_params = None
        best_set_left = []
        best_set_right = []

        for params in param_cadidates:

            left_set, right_set = [], []
            for (p, m) in data:
                output = self.evalFunction(p, params) 
                # TODO: die Funktion evalFunction erwartet beim erstellen der features
                # die zugehörige depth map als 2D Matrix und das ganze Bild als 3D RGB 
                if (output == True):
                    right_set.append((p, m))
                else:
                    left_set.append((p, m))

            Q = calculate_Q(right_set, left_set)

            if (Q > best_score_Q):
                best_score_Q = Q
                best_params = params
                best_set_left = left_set
                best_set_right = right_set

        self.params = best_params
        

class RegressionTree:
    def __init__(self, evalFunction, maxDepth):
        self.root = Node(evalFunction)
        self.evalFunction = evalFunction
        self.maxDepth = maxDepth
        self.isTrained = False

    def evaulate(self, p):
        if not self.isTrained:
            raise Exception('Error: Tree is not trained yet!')
        return self.root.evaluate(p)

    def train(self, data):
        [images, depth_map, pose_matrices] = data
        # TODO: we need depthmap, must pass to node
        
        samples = []
        for i in tqdm(range(len(images))):
            image = images[i]
            pose = pose_matrices[i]
            samples.append(get_sample_pixels(image.shape, depth_map, pose, num_samples=100))

        # TODO: start training by calling train at root node
       
        # self.isTrained = True
            ...


def get_sample_pixels (img_shape, depth_map, pose, num_samples):
    m, n = img_shape

    indices = np.random.choice(m * n, num_samples, replace=True)
    p = [(c % m, c / m) for c in indices]
    p_ext = [(x, y, depth_map[x, y], 1) for (x, y) in p]

    m = (pose @ p_ext)[:,:-1]
    # TODO: make sure this actually works
    return p, m


class FeatureType(Enum):
    """
    This bass class represents the three different feature types
    including 'Depth', 'Depth-Adaptive RGB' and 'Depth-Adaptive RGB + Depth'
    """
    DEPTH = 1
    DA_RGB = 2
    DA_RGB_DEPTH = 3

    
def get_feature(p: np.array, depth_map: np.array, img: np.array, params:any, type=FeatureType.DEPTH):
    """
    For a pixel coordinate p, corresponding image and depth map 
    as well as regression parameters obtain an image feature

    Parameters
    ----------
    p: np.array 
        containing the pixel coordinates (x,y)
    depth_map: np.array
        the images depth values in millimeters
    img: np.array
        the image as rgb 
    params: 
        parameters defined for each node
    fill_value: float
        fill invalid depth values with distance in millimeters

    Returns
    ----------
    feature value for pixel at position p
    or None if invalid depth value encountered
    """
    (tau, delta1, delta2, c1, c2, z) = params

    # check if the depth value is defined for initial pixel coordinate p
    if is_valid_depth(depth_map, p):
        p1 = p + delta1/depth_map[p[1], p[0]] 
        p2 = p + delta2/depth_map[p[1], p[0]] 
    else:
        return None

    # make sure to use integer coordinates
    p1 = p1.astype(int)
    p2 = p2.astype(int)

    if type is FeatureType.DEPTH:
        if is_valid_depth(depth_map, p1) and is_valid_depth(depth_map, p2):
            d1 = depth_map[p1[1],p1[0]] 
            d2 = depth_map[p2[1],p2[0]]
            return d1 - d2
        else:
            return None

    elif type is FeatureType.DA_RGB:
        if in_bounds(depth_map.shape, p1) and in_bounds(depth_map.shape, p2):
            i1 = img[p1[1], p1[0], c1] 
            i2 = img[p2[1], p2[0], c2]
            return i1 - i2
        else:
            return None

    elif type is FeatureType.DA_RGB_DEPTH:
        raise NotImplementedError()
   

def is_valid_depth(depth_map: np.array, p: np.array, invalid_value=65535) -> bool:
    """
    Check if a pixel lookup is valid by checking for invalid depth value
    and if the pixel coordinate lays inside the image boundary
    """
    if depth_map[p[1], p[0]] == invalid_value or not in_bounds(depth_map.shape, p):
        return True
    else:
        return False


def in_bounds(matrix_shape: tuple, index: np.array) -> bool:
    """
    Check if a given index is in the range of a matrix boundary
    by determining if its value is below zero or above the image size
    """
    if index[0] < 0 or index[1] < 0 or index[0] > matrix_shape[1] or index[1] > matrix_shape[0]:
        return True
    else:
        return False
