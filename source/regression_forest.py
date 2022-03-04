from collections.abc import Callable
from msilib.schema import Feature
from typing import Tuple

import numpy as np
import tqdm

from feature_extractor import FeatureExtractor, FeatureType
from data_holer import DataHolder, SampleHolder, Sample

def objective_reduction_in_variance (
    set_complete: DataHolder,
    set_left: DataHolder,
    set_right: DataHolder
    ) -> float:
    """
    Tree training objective function. Evaluates the fitness of the split according to the
    reduction in variance criterium (see 2.4 forest training)

    Parameters
    ----------
    set_complete: DataHolder
        The complete set before the tree node
    set_left: DataHolder
        The set, which got split to the left
    set_right: DataHolder
        The set, which got split to the right
    """
    (p_complete, w_complete) = set_complete.get_all_sample_data_points()
    (p_left, w_left) = set_left.get_all_sample_data_points()
    (p_right, w_right) = set_right.get_all_sample_data_points()

    num_samples = len(w_complete)
    fac_left = len(w_left) / num_samples
    fac_right = len(w_right) / num_samples

    return np.var(np.linalg.norm(w_complete)) \
        - fac_left * np.var(np.linalg.norm(w_left, axis=1)) \
        - fac_right * np.var(np.linalg.norm(w_right, axis=1))


"""
TOOD: Idea (by Vincenco):

Use "pre pruning" on the tree. Maybe use shannon entropy in objective function
to access the information gain at a node given a certain parameter set
"""

class Node:
    # TODO: Think about this! Might want to decouple feature function from image data after all?
    def __init__(self, feature_function: Callable[[np.array, np.array], bool], initial_params = any):
        """
        Create a new node with a given feature_function and initial parameters

        Parameters
        ----------
        feature_function: Callable[[np.array, np.array], bool]
            The feature function used to in this node
        initial_params: any
            The initial parameters for this node. Used for loading
            trained tree
        """
        self.feature_function = feature_function
        
        self.params = initial_params
        self.leared_response = None
        
        self.left_child = None
        self.right_child = None

    def is_leaf(self):
        return (self.left_child is None) and (self.right_child is None)

    def evaluate(self, p: np.array) -> any:
        """
        Evaluate the tree recursively starting at this node.

        Parameters
        ----------
        p: np.array
            Input feature (pixel coordinates)
        
        Returns
        -------
        any
            The response at the leaf node. Evaluated recursively

        """
        output = self.feature_function(self.params, p)
        if self.is_leaf():
            return self.leared_response

        nextNode = self.right_child if output else self.left_child
        return nextNode.evaulate(p)

    def train(self,
        depth: int,
        data: np.array,
        objective_function: Callable[[np.array, np.array], bool],
        param_sampler: Callable[[int], np.array],
        num_param_samples: int,
        max_depth: int
        ) -> None:
        """
        Train this node on the data using the objective_function

        Parameters
        ----------
        depth: int
            The current tree depth. Used for constraint on tree depth

        data: np.array
            The data used to train the node. Expected format:
            data = [[input_feature_0, ..1, ...], [target_response_0, ..1, ...]]

        num_param_samples: int
            The number of parameter samples to evaluate

        max_depth: int
            The maximum allowed depth of the tree. Use mode fitting if reached

        def objective_function(set_left: np.array, set_right: np.array) -> float:
            The objective function to min/maximise. It rates the quality of the
            split achived by this node (for a certian set of params).
            Parameters
            ----------
            set_left: np.array
                The left set after this node's split (formatted like data)
            set_right: np.array
                The right set after this node's split (formatted like data)
            Returns
            -------
            float judging the quality of this split
        
        def param_sampler(number: int) -> np.array
            The function used to sample from the parameter space for the node
            split parameters
            Parameters
            ----------
            number: int
                The number of parameter samples to generate
            Returns
            -------
            np.array of the paramters

        """

        param_split_masks = []

        feature_function_vec = np.vectorize(self.feature_function, excluded=['p'])
        def evalulate_data_with_params (params: np.array) -> float:
            """
            Evaluate a single set of parameters on the whole data-set.
            Returns the objective-functions judgement
            """
            param_split_mask = feature_function_vec(input_features, params)
            param_split_masks.append(param_split_mask)

            left_split_features = input_features[param_split_mask]
            right_split_features = input_features[param_split_mask == False]
            left_split_responses = target_responses[param_split_mask]
            right_split_responses = target_responses[param_split_mask == False]

            set_left = np.array([left_split_features, left_split_responses])
            set_right = np.array([right_split_features, right_split_responses])
            return objective_function(set_left, set_right)

        input_features = data[0]
        target_responses = data[1]

        if len(input_features) == 0:
            # This node only received a set of size one. It should be a leaf node
            self.left_child = None
            self.right_child = None
            self.leared_response = target_responses[0]
            return

        if depth == max_depth:
            # This node reached maximum tree depth. It should be a leaf node
            self.left_child = None
            self.right_child = None
            # TODO: Implement mode fitting (mean filtering?) on target_responses
            self.leared_response = np.mean(target_responses, axis=0)
            return    

        evalulate_data_with_params_vec = np.vectorize(evalulate_data_with_params)
        param_samples = param_sampler(num_param_samples)
        param_scores = evalulate_data_with_params_vec(param_samples)

        best_param_index = np.argmax(param_scores)
        best_params = param_samples[best_param_index]
        best_split_mask = param_split_masks[best_param_index]

        # Yes, this calculation is done twice. Should be worth the memory saved by only remebering the mask,
        # instaed of keeping copies of the complete feature and response sets
        # Could be avoided by keeping running best_{set_left, set_right, params} varaibles. This might be
        # a problem for parallelization though.
        best_set_left = (input_features[best_split_mask], target_responses[best_split_mask])
        best_set_right = (input_features[best_split_mask == False], target_responses[best_split_mask == False])

        self.params = best_params
        self.left_child = Node(self.feature_function)
        self.right_child = Node(self.feature_function)

        self.left_child.train(depth + 1, best_set_left, objective_function, param_sampler, num_param_samples, max_depth)
        self.right_child.train(depth + 1, best_set_right, objective_function, param_sampler, num_param_samples, max_depth)

                

"""
Vincenco's (some friend) idea:

google "pre pruning"
use shannon entropy in objective function?
"""
class RegressionTree:
    def __init__(self, feature_function, max_depth: int):
        self.root = Node(feature_function)
        self.feature_function = feature_function
        self.max_depth = max_depth
        self.is_trained = False

    def evaulate(self, p):
        if not self.is_trained:
            raise Exception('Error: Tree is not trained yet!')
        return self.root.evaluate(p)

    def train(self, data):

        """
        TODO:
        * FeatureExtractor currently acts as a holder for data from one image (pose, rgb, depth).
          Can be used to generate (input_feature, target_respose) sample pairs. Might need to pass
          the feature_extractor instance for each set of (num_samples) samples into the nodes for
          training (where feature_function needs to be evaluated) and evaluation.
          
        * Whole tree training. Depends on FeatureExtractor reference handling.
        * Plant a whole forest.
        """


        # TODO: Cleanup
        samples = []
        for i in tqdm(range(len(images))):
            image = images[i]
            pose = pose_matrices[i]
            samples.append(get_sample_pixels(image.shape, depth_map, pose, num_samples=100))

        # start training by calling train at root node
        # self.is_trained = True
