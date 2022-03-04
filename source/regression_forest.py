from collections.abc import Callable
from typing import Tuple, List

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
        self.best_score = 0
        self.best_set_right = None
        self.best_set_left = None
        self.leared_response = None
        
        self.left_child = None
        self.right_child = None

    def is_leaf(self):
        return (self.left_child is None) and (self.right_child is None)

    def evaluate(self, samples: Tuple[SampleHolder, Sample]) -> List[any]:
        """
        Evaluate the tree recursively starting at this node.

        Parameters
        ----------
        samples: SampleHolder
            Input samples (pixel coorindates)
        samples: Sample
            A single input sample (pixel coordinates)
        
        Returns
        -------
        any
            The response at the leaf node. Evaluated recursively

        """
        if self.is_leaf():
            return self.leared_response

        if isinstance(samples, Sample):
            samples = SampleHolder(samples.feature_extractor, [samples.p], [samples.w])

        outputs = samples.get_features_with_parameters(self.params, self.feature_type)
        for output in outputs:
            nextNode = self.right_child if output else self.left_child
            return nextNode.evaulate(samples)

    def train(self, 
        depth: int,
        data: DataHolder,
        objective_function: Callable[[DataHolder, DataHolder, DataHolder], float],
        param_sampler: Callable[[int], np.array],
        num_param_samples: int,
        max_depth: int,
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
            self.best_score = 0
            self.leared_response = None
            self.left_child = None
            self.right_child = None

        if depth == max_depth or len(data) == 1:
            # This should be a leaf node
            # We need to find the "mode"
            (_, w_s) = data.get_all_sample_data_points()
            self.leared_response = np.mean(w_s, axis=0)
            self.left_child = None
            self.right_child = None
            return
        else:
            self.left_child = Node(self.feature_type, None) if self.left_child == None else self.left_child
            self.right_child = Node(self.feature_type, None) if self.right_child == None else self.right_child
 
        param_samples = param_sampler(num_param_samples)
        for param_sample in param_samples:
            split_mask = data.get_features_with_parameters(param_sample, self.feature_type)
            split_mask_inverted = [mask == False for mask in split_mask]
            set_left = data[split_mask]
            set_right = data[split_mask_inverted]
            score = objective_function(data, set_left, set_right)

            if score > self.best_score:
                self.best_score = score
                self.params = param_sample
                self.best_set_left = set_left
                self.best_set_right = set_right

        self.left_child.train(depth + 1, self.best_set_left, objective_function,
                              param_sampler, num_param_samples, max_depth)
        self.right_child.train(depth + 1, self.best_set_right, objective_function,
                              param_sampler, num_param_samples, max_depth)                

class RegressionTree:
    def __init__(self,
        max_depth: int,
        feature_type: FeatureType,
        param_sampler: Callable[[int], np.array],
        objective_function: Callable[[DataHolder, DataHolder, DataHolder], float]):

        self.root = Node()
        self.is_trained = False
        self.max_depth = max_depth
        self.feature_type = feature_type
        self.param_sampler = param_sampler
        self.objective_function = objective_function

    def evaulate(self, samples: Tuple[SampleHolder, Sample]) -> List[any]:
        if not self.is_trained:
            raise Exception('Error: Tree is not trained yet!')
        return self.root.evaluate(samples)

    def train(self, data: DataHolder, num_param_samples: int, reset: bool = False):
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
        
        self.root.train(
            depth = 0,
            data = data,
            objective_function = self.objective_function,
            param_sampler = self.param_sampler,
            num_param_samples = num_param_samples,
            max_depth = self.max_depth,
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

    def train(self, data: DataHolder, num_param_samples: int, reset: bool = False):
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
        for tree in self.trees:
            tree.train(data, num_param_samples, reset)
            