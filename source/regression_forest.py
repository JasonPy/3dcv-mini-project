from collections.abc import Callable
from typing import Tuple, List

import numpy as np
from tqdm import tqdm
print = tqdm.write

from feature_extractor import FeatureExtractor, FeatureType
from data_holder import DataHolder, ProcessingPool, DataHolder
from utils import millis

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
    var_left = 0 if fac_left == 0 else np.var(np.linalg.norm(w_left, axis=1))
    var_right = 0 if fac_right == 0 else np.var(np.linalg.norm(w_right, axis=1))

    return np.var(np.linalg.norm(w_complete)) - fac_left * var_left - fac_right * var_right


"""
TOOD: Idea (by Vincenco):

Use "pre pruning" on the tree. Maybe use shannon entropy in objective function
to access the information gain at a node given a certain parameter set
"""

class Node:
    def __init__(self, feature_type: FeatureType, initial_params = any):
        """
        Create a new node with a given feature_function and initial parameters

        Parameters
        ----------
        feature_type: FeatureType
            The feature type used to in this node
        initial_params: any
            The initial parameters for this node. Used for loading
            trained tree
        """
        self.feature_type = feature_type
        
        self.params = initial_params
        self.best_score = -np.inf
        self.best_set_right = None
        self.best_set_left = None
        self.leared_response = None
        
        self.left_child = None
        self.right_child = None

    def is_leaf(self):
        return (self.left_child is None) and (self.right_child is None)

    def evaluate(self, samples: DataHolder):
        """
        Evaluate the tree recursively starting at this node.

        Parameters
        ----------
        samples: DataHolder
            Input samples (pixel coorindates)
        
        Returns
        -------
        any
            The response at the leaf node. Evaluated recursively

        """
        if self.is_leaf():
            return self.leared_response

        outputs = samples.get_features_with_parameters(self.params, self.feature_type)
        for output in outputs:
            nextNode = self.right_child if output else self.left_child
            return nextNode.evaulate(samples)

    def train(self, 
        depth: int,
        data: DataHolder,
        processing_pool: ProcessingPool,
        objective_function: Callable[[DataHolder, DataHolder, DataHolder], float],
        param_sampler: Callable[[int], np.array],
        num_param_samples: int,
        max_depth: int,
        _tqdm: tqdm,
        reset: bool = False
        ) -> None:
        """
        Train this node on the data using the objective_function

        Parameters
        ----------
        depth: int
            The current tree depth. Used for constraint on tree depth

        max_depth: int
            The maximum allowed depth of the tree. Use mode fitting if reached

        data: DataHolder
            The data used to train the node. Expected format:
            data = [[input_feature_0, ..1, ...], [target_response_0, ..1, ...]]

        num_param_samples: int
            The number of parameter samples to evaluate

        reset: bool
            Reset the maximum score which had been achieved so far

        objective_function: Callable
            The objective function to min/maximise. It rates the quality of the
            split achived by this node (for a certian set of params).

            set_complete: DataHolder
                The right set after this node's split (formatted like data)

            set_left: np.array, set_right: np.array
                The left set after this node's split (formatted like data)
                
            Returns
            -------
            float judging the quality of this split
        
        param_sampler: Callable
            The function used to sample from the parameter space for the node
            split parameters

            number: int
                The number of parameter samples to generate
            Returns
            -------
            np.array of the paramters

        """

        if reset:
            self.best_score = -np.inf
            self.leared_response = None
            self.left_child = None
            self.right_child = None

        _len_data = len(data)
        if _len_data == 0:
            return

        is_leaf_node = False

        if depth == max_depth or _len_data == 1:
            # This should be a leaf node
            # We need to find the "mode"
            is_leaf_node = True
            _tqdm.update(_len_data)
            (_, w_s) = data.get_all_sample_data_points()
            self.leared_response = np.mean(w_s, axis=0)
            self.left_child = None
            self.right_child = None
        else:
            self.left_child = Node(self.feature_type, None) if self.left_child == None else self.left_child
            self.right_child = Node(self.feature_type, None) if self.right_child == None else self.right_child
 
        if not is_leaf_node:
            _ms_start = millis()

            param_samples = param_sampler(num_param_samples)
            masks = data.get_features_with_parameters(
                processing_pool = processing_pool,
                param_samples = param_samples,
                feature_type = self.feature_type)

            _delta_get_features = millis() - _ms_start

            for param_sample, mask in zip(param_samples, masks):
                set_left = data[mask]
                set_right = data[mask == False]
                score = objective_function(set_complete = data, set_left = set_left, set_right = set_right)
                if score > self.best_score:
                    self.best_score = score
                    self.params = param_sample
                    self.best_set_left = set_left
                    self.best_set_right = set_right

            _tqdm.update(_len_data)
            _len_left = len(self.best_set_left)
            _len_right = len(self.best_set_right)

            _delta_split_and_score = millis() - _delta_get_features - _ms_start
            _str_features = f'{_len_data * num_param_samples:20} samples evaluated in {_delta_get_features:8.0F}ms'
            _str_split = f'Split {_len_data:10} into {_len_left:10} and {_len_right:10} in {_delta_split_and_score:8.0F}ms'
            kilo_it_per_sec_str = f'{(_len_data * num_param_samples) / (_delta_get_features + _delta_split_and_score):.1F}'
            print(f'Node trained: {_str_split} | {_str_features} ------> {kilo_it_per_sec_str:6}Kit/s {("-" * (depth + 1))}')

            if _len_left == 0:
                self.left_child = None
                self.right_child = None
                self.leared_response = np.mean(self.best_set_right.get_all_sample_data_points()[1], axis=0)
                is_leaf_node = True
            elif _len_right == 0:
                self.left_child = None
                self.right_child = None
                self.leared_response = np.mean(self.best_set_left.get_all_sample_data_points()[1], axis=0)
                is_leaf_node = True
            else:
                self.left_child.train(
                    depth = depth + 1,
                    data = self.best_set_left,
                    processing_pool = processing_pool,
                    objective_function = objective_function,
                    param_sampler = param_sampler,
                    num_param_samples = num_param_samples,
                    max_depth = max_depth,
                    _tqdm = _tqdm,
                    reset = reset)
                self.right_child.train(
                    depth = depth + 1,
                    data = self.best_set_right,
                    processing_pool = processing_pool,
                    objective_function = objective_function,
                    param_sampler = param_sampler,
                    num_param_samples = num_param_samples,
                    max_depth = max_depth,
                    _tqdm = _tqdm,
                    reset = reset)

        if is_leaf_node:
            print(f'Leaf node created at depth {depth}')
            _tqdm.update(_len_data * (max_depth - depth - 1))

class RegressionTree:
    def __init__(self,
        max_depth: int,
        feature_type: FeatureType,
        param_sampler: Callable[[int], np.array],
        objective_function: Callable[[DataHolder, DataHolder, DataHolder], float]):

        self.root = Node(feature_type=feature_type)
        self.is_trained = False
        self.max_depth = max_depth
        self.feature_type = feature_type
        self.param_sampler = param_sampler
        self.objective_function = objective_function

    def evaulate(self, samples: DataHolder):
        if not self.is_trained:
            raise Exception('Error: Tree is not trained yet!')
        return self.root.evaluate(samples)

    def train(self, data: DataHolder, processing_pool: ProcessingPool, num_param_samples: int, reset: bool = False):
        """
        Train this tree with the list of FeatureExtractors (images) given.
        
        Parameters
        ----------
        data: DataHolder
            The data to train this tree with
        num_param_samples: int
            The number of parameters samples to train
        reset: bool = False
            Reset tree
        """
        
        _tqdm = tqdm(
            iterable = None,
            desc = 'Training tree  ',
            total = len(data) * self.max_depth,
            ascii = True
        )

        self.root.train(
            depth = 0,
            data = data,
            processing_pool = processing_pool,
            objective_function = self.objective_function,
            param_sampler = self.param_sampler,
            num_param_samples = num_param_samples,
            max_depth = self.max_depth,
            _tqdm = _tqdm,
            reset = reset)
            
        self.is_trained = True

class RegressionForest:
    def __init__(self,
        num_trees: int,
        max_depth: int,
        feature_type: FeatureType,
        param_sampler: Callable[[int], np.array],
        objective_function: Callable[[DataHolder, DataHolder, DataHolder], float]):
        self.is_trained = False
        self.trees = [RegressionTree(max_depth, feature_type, param_sampler, objective_function) for _ in range(num_trees)]

    def train(self, data: DataHolder, processing_pool: ProcessingPool, num_param_samples: int, reset: bool = False):
        """
        Train this forest with the list of FeatureExtractors (images) given.

        Parameters
        ----------
        data: DataHolder
            The data to train this forest with
        num_param_samples: int
            The number of parameters samples to train
        reset: bool = False
            Reset tree
        """
        for tree in tqdm(self.trees, ascii = True, desc = f'Training forest'):
            tree.train(data, processing_pool, num_param_samples, reset)
            