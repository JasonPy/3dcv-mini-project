# import modules
import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def pixelRGBDepthFunction (params, p_array):
    (tau, delta1, delta2, c1, c2, z) = params
    (x, y) = p
    return x * c2 < tau

def getSamplePixels (image, pose, num_samples):
    (m, n, _) = image.shape

    indices = np.random.choice(m * n, num_samples, replace=True)
    p = [(c % m, c / m) for c in indices]
    p_ext = [(x, y, image[x, y, 3], 1) for (x, y) in p]

    m = (pose @ p_ext)[:,:-1]
    # TODO: make sure this actually works
    return (p, m)

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
        data = [[p, m], [...]]

        param_cadidates = []
        # sample

        best_score_Q = 0
        best_params = None
        best_set_left = []
        best_set_right = []

        for params in param_cadidates:

            left_set, right_set = [], []
            for (p, m) in data:
                output = self.evalFunction(params, p)
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

    def build_and_train_tree(self, data):
        [images, pose_matrices] = data
        
        samples = []
        for i in tqdm(range(len(images))):
            image = images[i]
            pose = pose_matrices[i]
            samples.append(getSamplePixels(image, pose, num_samples=100))

        if (iters > self.maxDepth) or (self.rig)
            # make leaf node

            # split into left, right
            # save params, substes in self.rightChild


        self.isTrained = True

