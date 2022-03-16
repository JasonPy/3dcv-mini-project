from typing import Callable
from multiprocessing import cpu_count
from typing import Tuple

from numba import njit
import numpy as np
from numpy.random import choice, uniform
from tqdm import tqdm
write = tqdm.write

from feature_extractor import FeatureType, get_features_for_samples
from processing_pool import ProcessingPool
from utils import get_mode, millis, vector_3d_array_variance, split_set

@njit
def param_sampler(num_samples: int) -> np.array:
    rgb_coords = np.array([0, 1, 2])
    tau = np.zeros(num_samples)
    delta1x = uniform(-130 * 100, 130 * 100, num_samples)
    delta1y = uniform(-130 * 100, 130 * 100, num_samples)
    delta2x = uniform(-130 * 100, 130 * 100, num_samples)
    delta2y = uniform(-130 * 100, 130 * 100, num_samples)
    c1 = choice(rgb_coords, num_samples, replace=True)
    c2 = choice(rgb_coords, num_samples, replace=True)
    return np.stack((tau, delta1x, delta1y, delta2x, delta2y, c1, c2)).T

@njit
def objective_reduction_in_variance (
    w_complete: np.array,
    w_left: np.array,
    w_right: np.array
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

    num_samples = len(w_complete)
    if num_samples == 0:
        return -np.inf
    fac_left = len(w_left) / num_samples
    fac_right = len(w_right) / num_samples
    var_left = 0 if fac_left == 0 else vector_3d_array_variance(w_left)
    var_right = 0 if fac_right == 0 else vector_3d_array_variance(w_right)

    return vector_3d_array_variance(w_complete) - fac_left * var_left - fac_right * var_right

@njit
def regression_tree_worker_score(image_data, p_s, w_s, param_sample, objective_function, feature_type):
    mask_valid, mask_split = get_features_for_samples(image_data, p_s, param_sample, feature_type)
    w_s_valid = w_s[mask_valid]
    set_left, set_right = split_set(w_s_valid, mask_split)
    score = objective_function(w_complete = w_s_valid, w_left = set_left, w_right = set_right)
    return score

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
        self.leared_response = None
        
        self.left_child = None
        self.right_child = None

    def is_leaf(self):
        return (self.left_child is None) and (self.right_child is None)

    def evaluate(self, samples: np.array, feature_type: FeatureType, processing_pool: ProcessingPool):
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
        if (len(samples) == 0):
            return samples

        if self.is_leaf():
            return np.full((len(samples), self.leared_response.shape[0]), self.leared_response)
        else:
            outputs = np.full((len(samples), 3), -np.inf)
            image_data, close_shm = processing_pool.get_image_data()
            mask_valid, mask_split = get_features_for_samples(image_data, samples, self.params, feature_type)
            close_shm()
            image_data = None
            samples_valid, _ = split_set(samples, mask_valid)
            split_left, split_right = split_set(samples_valid, mask_split)
            
            response_left = self.left_child.evaluate(split_left, feature_type, processing_pool)
            response_right = self.right_child.evaluate(split_right, feature_type, processing_pool)
            outputs_masked = outputs[mask_valid].copy()
            outputs_masked[mask_split] = response_left
            outputs_masked[~mask_split] = response_right
            outputs[mask_valid] = outputs_masked
            return outputs

    def train(self, 
        depth: int,
        data: Tuple[np.array, np.array],
        processing_pool: ProcessingPool,
        param_sampler: Callable[[int], np.array],
        num_param_samples: int,
        max_depth: int,
        _tqdm: tqdm,
        label: str = '',
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

        p_s, w_s = data
        _len_data = len(p_s)
        if _len_data == 0:
            return

        is_leaf_node = False
        if depth == max_depth or _len_data == 1:
            # This should be a leaf node
            # We need to find the "mode"
            is_leaf_node = True
            _tqdm.update(_len_data)
            self.leared_response = get_mode(w_s)
            self.left_child = None
            self.right_child = None
        else:
            self.left_child = Node(self.feature_type, None) if self.left_child == None else self.left_child
            self.right_child = Node(self.feature_type, None) if self.right_child == None else self.right_child
 
        if not is_leaf_node:
            _ms_start = millis()

            param_samples = param_sampler(num_param_samples)
            results = []

            if (_len_data < 350) or False:
                image_data, close_shm = processing_pool.get_image_data()
                objective_function, feature_type = processing_pool.get_worker_params()
                for param_sample in param_samples:
                    results.append(regression_tree_worker_score(
                        image_data = image_data,
                        p_s = p_s,
                        w_s = w_s,
                        param_sample = param_sample,
                        objective_function = objective_function,
                        feature_type = feature_type
                    ))
                image_data = None
                close_shm()
            else:
                work_datas = [(p_s, w_s, param_sample) for param_sample in param_samples]
                results = processing_pool.process_work(work_datas)

            _delta_get_features = millis() - _ms_start

            index_best_sample = np.argmax(results)
            score = results[index_best_sample]
            self.params = param_samples[index_best_sample]
            self.best_score = score

            image_data, close_shm = processing_pool.get_image_data()
            objective_function, feature_type = processing_pool.get_worker_params()
            mask_valid, mask_split = get_features_for_samples(image_data, p_s, self.params, feature_type)
            image_data = None
            close_shm()

            best_len_invalid = sum(~mask_valid)

            best_p_s_valid, _ = split_set(p_s, mask_valid)
            best_w_s_valid, _ = split_set(w_s, mask_valid)
            best_p_s_left, best_p_s_right = split_set(best_p_s_valid, mask_split)
            best_w_s_left, best_w_s_right = split_set(best_w_s_valid, mask_split)

            results = None
            param_samples = None
            _tqdm.update(_len_data)
            _len_left = len(best_p_s_left)
            _len_right = len(best_p_s_right)

            _delta_split_and_score = millis() - _delta_get_features - _ms_start
            _str_split = f'| {_len_data:10} in | {best_len_invalid:8} inval | {_len_left:8} left | {_len_right:8} right | {_delta_split_and_score:4.0F}ms split |'
            _str_features = f'{_len_data * num_param_samples:13} samples | {_delta_get_features:8.0F}ms eval |'
            kilo_it_per_sec_str = f'{(_len_data * num_param_samples) / (_delta_get_features + _delta_split_and_score):.1F}'
            write(f'Node trained         {_str_split} {_str_features} {kilo_it_per_sec_str:7}Kit/s | {label:16} id |')

            if best_len_invalid == _len_data:
                self.left_child = None
                self.right_child = None
                self.leared_response = get_mode(best_w_s_valid)
                is_leaf_node = True
            elif _len_left == 0:
                self.left_child = None
                self.right_child = None
                self.leared_response = get_mode(best_w_s_valid)
                is_leaf_node = True
            elif _len_right == 0:
                self.left_child = None
                self.right_child = None
                self.leared_response = get_mode(best_w_s_valid)
                is_leaf_node = True
            else:
                p_s = None
                w_s = None
                _tqdm.update(best_len_invalid * (max_depth - depth - 1))
                self.left_child.train(
                    depth = depth + 1,
                    data = (best_p_s_left, best_w_s_left),
                    processing_pool = processing_pool,
                    param_sampler = param_sampler,
                    num_param_samples = num_param_samples,
                    max_depth = max_depth,
                    _tqdm = _tqdm,
                    label = f'{label}0',
                    reset = reset)
                self.right_child.train(
                    depth = depth + 1,
                    data = (best_p_s_right, best_w_s_right),
                    processing_pool = processing_pool,
                    param_sampler = param_sampler,
                    num_param_samples = num_param_samples,
                    max_depth = max_depth,
                    _tqdm = _tqdm,
                    label = f'{label}1',
                    reset = reset)

        if is_leaf_node:
            _tqdm.update(_len_data * (max_depth - depth - 1))

class RegressionTree:
    def __init__(self,
        max_depth: int,
        feature_type: FeatureType,
        param_sampler: Callable[[int], np.array],
        objective_function: Callable[[np.array, np.array, np.array], float]):

        self.root = Node(feature_type=feature_type)
        self.is_trained = False
        self.max_depth = max_depth
        self.feature_type = feature_type
        self.param_sampler = param_sampler
        self.objective_function = objective_function

    def evaulate(self, samples: np.array, images_data: Tuple[np.array, np.array, np.array]):
        if not self.is_trained:
            raise Exception('Error: Tree is not trained yet!')

        processing_pool = ProcessingPool(images_data)
        results = self.root.evaluate(samples, self.feature_type, processing_pool)
        processing_pool.stop_workers()
        return results

    def train(self, 
        images_data: Tuple[np.array, np.array, np.array],
        data: Tuple[np.array, np.array],
        num_param_samples: int,
        reset: bool = False,
        num_workers: int = cpu_count()):
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

        processing_pool = ProcessingPool(
            images_data = images_data,
            num_workers = num_workers,
            worker_function = regression_tree_worker_score,
            worker_params = (self.objective_function, self.feature_type))
        
        try:
            images_data = None
            tqdm.write(f'Training forest with {len(data[0])} samples')
        
            _tqdm = tqdm(
                iterable = None,
                desc = 'Training tree  ',
                total = len(data[0]) * self.max_depth,
                ascii = True
            )

            self.root.train(
                depth = 0,
                data = data,
                processing_pool = processing_pool,
                param_sampler = self.param_sampler,
                num_param_samples = num_param_samples,
                max_depth = self.max_depth,
                _tqdm = _tqdm,
                reset = reset)
                
            self.is_trained = True
        except KeyboardInterrupt:
            tqdm.write(f'Stopping training due to KeyboardInterrupt')
        finally:
            processing_pool.stop_workers()

class RegressionForest:
    def __init__(self,
        num_trees: int,
        max_depth: int,
        feature_type: FeatureType,
        param_sampler: Callable[[int], np.array],
        objective_function: Callable[[np.array, np.array, np.array], float]):
        self.is_trained = False
        self.trees = [RegressionTree(max_depth, feature_type, param_sampler, objective_function) for _ in range(num_trees)]

    def evaluate(self, samples: np.array, images_data: Tuple[np.array, np.array, np.array]):
        if not self.is_trained:
            raise Exception('Forest is not trained!')

        return self.trees[0].evaulate(samples, images_data)

    def train(self,
        data: Tuple[np.array, np.array],
        images_data: Tuple[np.array, np.array, np.array],
        num_param_samples: int,
        reset: bool = False,
        num_workers: int = cpu_count()):
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
            tree.train(images_data, data, num_param_samples, reset, num_workers)

        self.is_trained = True
            