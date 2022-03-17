from multiprocessing.dummy import JoinableQueue
from typing import Callable
from multiprocessing import cpu_count, Lock
from typing import Tuple

from numba import njit, jit, prange, types, typeof
from numba.experimental import jitclass
import numpy as np
from numpy.random import choice, uniform
from tqdm import tqdm
write = tqdm.write

from feature_extractor import FeatureType, get_features_for_samples
from processing_pool import ProcessingPool
from utils import get_mode, millis, vector_3d_array_variance, split_set
from data_loader import DataLoader

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

@njit()
def calculate_score_for_params(image_data, p_s, w_s, param_sample, objective_function, feature_type):
    mask_valid, mask_split = get_features_for_samples(image_data, p_s, param_sample, feature_type)
    w_s_valid = w_s[mask_valid]
    set_left, set_right = split_set(w_s_valid, mask_split)
    score = objective_function(w_complete = w_s_valid, w_left = set_left, w_right = set_right)
    return score

@jitclass([('node_id', types.unicode_type),
           ('is_leaf', types.boolean),
           ('response', types.float64[:]),
           ('params', types.float64[:]),
           ('p_s_left', types.float64[:,:]),
           ('w_s_left', types.float64[:,:]),
           ('p_s_right', types.float64[:,:]),
           ('w_s_right', types.float64[:,:])])
class TreeWorkerResult:
    def __init__(self,
        node_id: str,
        is_leaf: bool,
        response: np.array,
        params: np.array = np.array([], np.float64),
        p_s_left: np.array = np.array([[]], np.float64),
        w_s_left: np.array = np.array([[]], np.float64),
        p_s_right: np.array = np.array([[]], np.float64),
        w_s_right: np.array = np.array([[]], np.float64)):
        self.node_id = node_id
        self.is_leaf = is_leaf
        self.response = response
        self.params = params
        self.p_s_left = p_s_left
        self.w_s_left = w_s_left
        self.p_s_right = p_s_right
        self.w_s_right = w_s_right

@njit
def handle_training_progress(*whatever):
    pass

@jit(nopython = True, parallel = True)
def regression_tree_worker(image_data, work_data, worker_params):
    # Extract work data
    num_param_samples, max_depth, feature_type, param_sampler, objective_function = worker_params
    node_id, depth, p_s, w_s = work_data

    # Auxiliary local variables
    tree_levels_below = max_depth - depth - 1
    len_data = len(p_s)
    is_leaf_node = False
    # _ms_start = millis()

    # Check if this node should not be trained
    if len_data == 1 or depth == max_depth:
        is_leaf_node = True
        handle_training_progress(len_data)
        result = TreeWorkerResult(node_id, True, get_mode(w_s))

    if not is_leaf_node:
        # Generate parameter samples
        param_samples = param_sampler(num_param_samples)
        scores = np.ndarray((num_param_samples), dtype=np.float64)

        # Calculate scores for all parameter samples
        for i in prange(num_param_samples):
            scores[i] = calculate_score_for_params(
                image_data = image_data,
                p_s = p_s,
                w_s = w_s,
                param_sample = param_samples[i],
                objective_function = objective_function,
                feature_type = feature_type)
        # _delta_get_features = millis() - _ms_start
        
        # Find best parameter and calculate split (again, I know)
        max_score_index = np.argmax(scores)
        best_params = param_samples[max_score_index]
        mask_valid, mask_split = get_features_for_samples(image_data, p_s, best_params, feature_type)

        # Split the input data        
        w_s_valid = w_s[mask_valid]
        w_s_left, w_s_right = split_set(w_s_valid, mask_split)
        p_s_valid = p_s[mask_valid]
        p_s_left, p_s_right = split_set(p_s_valid, mask_split)

        # Calculate lenghts
        len_invalid = np.sum(~mask_valid)
        len_left = np.sum(mask_split)
        len_right = np.sum(~mask_split)

        # Report training progress
        # _delta_split = millis() - _delta_get_features - _ms_start
        # handle_training_progress(len_data, node_id, len_data, len_invalid, len_left, len_right, _delta_get_features, _delta_split)

        if len_invalid == len_data:
            # All samples are invalid. Edge case; not observed so far
            # handle_write(f'Error in node {node_id}: All samples considered invalid')
            result = TreeWorkerResult(
                node_id = node_id,
                is_leaf = True,
                response = w_s[0])# This is not correct.

        elif len_right == 0 or len_left == 0:
            # All samples are split to one side -> this should be a leaf node
            is_leaf_node = True
            result = TreeWorkerResult(
                node_id = node_id,
                is_leaf = True,
                response = get_mode(w_s_valid))

        else:
            # Report training progress on invalid nodes, trigger next node training
            handle_training_progress(len_invalid * tree_levels_below)
            result = TreeWorkerResult(
                node_id = node_id,
                is_leaf = False,
                params = best_params,
                p_s_left = p_s_left,
                w_s_left = w_s_left,
                p_s_right = p_s_right,
                w_s_right = w_s_right)

    if is_leaf_node:
        # Report training progress ("skipped" calculations since this is a leaf node)
        handle_training_progress(len_data * tree_levels_below)

    return result


"""
TOOD: Idea (by Vincenco):

Use "pre pruning" on the tree. Maybe use shannon entropy in objective function
to access the information gain at a node given a certain parameter set
"""

param_type = typeof(param_sampler(1)[0])
print(param_type)
spec = [('id', types.unicode_type),
        ('depth', types.int16),
        ('params', param_type),
        ('response', types.float64[:]),
        ('node_id_left', types.unicode_type),
        ('node_id_right', types.unicode_type)]
@jitclass(spec)
class Node:
    def __init__(self, node_id: str, depth: int = 0):
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

        self.id = node_id
        self.depth = 0
        self.params = np.array([0], dtype=np.float64)
        self.response = np.array([.0, .0])
        self.node_id_left = ''
        self.node_id_right = ''

    def is_leaf(self):
        return (self.node_id_left == '') and (self.node_id_right == '')

    def evaluate(self, tree: 'RegressionTree', samples: np.array, feature_type: FeatureType, processing_pool: ProcessingPool):
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
            
            left_child = tree.nodes[self.node_id_left]
            right_child = tree.nodes[self.node_id_right]

            response_left = left_child.evaluate(split_left, tree, feature_type, processing_pool)
            response_right = right_child.evaluate(split_right, tree, feature_type, processing_pool)
            outputs_masked = outputs[mask_valid].copy()
            outputs_masked[mask_split] = response_left
            outputs_masked[~mask_split] = response_right
            outputs[mask_valid] = outputs_masked
            return outputs

def tree_result_worker(tree: 'RegressionTree', queue_result: JoinableQueue):
    while (True):
        result = queue_result.get()
        tree.handle_node_result(result)
        queue_result.task_done()        

class RegressionTree:
    def __init__(self,
        max_depth: int,
        feature_type: FeatureType,
        param_sampler: Callable[[int], np.array],
        num_param_samples: int,
        objective_function: Callable[[np.array, np.array, np.array], float]):

        self.nodes = dict()
        self.is_trained = False
        self.max_depth = max_depth
        self.feature_type = feature_type
        self.param_sampler = param_sampler
        self.num_param_samples = num_param_samples
        self.objective_function = objective_function

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def ensure_training_in_progress(self):
        if self.training_lock == None:
            raise Exception('Training not in progress!')

    def handle_training_progress(self, num_evaluations: int, node_id: str, len_data, len_invalid, len_left, len_right, _delta_get_features, _delta_split):
        self.ensure_training_in_progress()
        with self.training_lock:
            self.tqdm.update(num_evaluations)

            # More elaborate logging
            if not node_id == None:
                _str_split = f'| {len_data:10} in | {len_invalid:8} inval | {len_left:8} left | {len_right:8} right | {_delta_split:4.0F}ms split |'
                _str_features = f'{len_data * self.num_param_samples:13} samples | {_delta_get_features:8.0F}ms eval |'
                kilo_it_per_sec_str = f'{(len_data * self.num_param_samples) / (_delta_get_features + _delta_split):.1F}'
                tqdm.write(f'Node trained         {_str_split} {_str_features} {kilo_it_per_sec_str:7}Kit/s | {node_id:16} id |')


    def handle_write(self, message: str):
        self.ensure_training_in_progress()
        with self.training_lock:
            tqdm.write(message)

    def handle_node_result(self, result: TreeWorkerResult):
        self.ensure_training_in_progress()

        node_id = result.node_id
        is_leaf = result.is_leaf
        response = result.response
        params = result.params
        p_s_left = result.p_s_left
        w_s_left = result.w_s_left
        p_s_right = result.p_s_right
        w_s_right = result.w_s_right
        
        with self.training_lock:
            if not node_id in self.nodes:
                raise Exception(f'Could not find node with id {node_id}')

            node = self.nodes[node_id]
            if is_leaf:
                node.response = response
            else:
                # Create new nodes
                node_left = Node(node.id + '0', node.depth + 1)
                node_right = Node(node.id + '1', node.depth + 1)
                self.add_node(node_left)
                self.add_node(node_right)

                # Save this node's training result
                node.params = params
                node.node_id_left = node_left.id
                node.node_id_right = node_right.id
                
                # Enqueue the new nodes for training
                train_left_work_data = (node_left.id, node_left.depth, p_s_left, w_s_left)
                train_right_work_data = (node_right.id, node_right.depth, p_s_right, w_s_right)
                self.processing_pool.enqueue_work_async(train_left_work_data)
                self.processing_pool.enqueue_work_async(train_right_work_data)

            # I don't really know if this is necessary. I want pointers :(
            self.nodes[node_id] = node
            
    def evaulate(self, samples: np.array, images_data: Tuple[np.array, np.array, np.array]):
        if not self.is_trained:
            raise Exception('Error: Tree is not trained yet!')

        processing_pool = ProcessingPool(images_data)
        results = self.nodes['0'].evaluate(samples, self.feature_type, processing_pool)
        processing_pool.stop_workers()
        return results

    def train(self, 
        data_loader: DataLoader,
        scene_name: str,
        train_indices: np.array,
        num_samples_per_images: int,
        num_workers: int):
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

        images_data = data_loader.load_dataset(scene_name = scene_name, image_indices = train_indices)
        data_samples = data_loader.sample_from_data_set(images_data = images_data, num_samples = num_samples_per_images)

        worker_params = self.num_param_samples, self.max_depth, self.feature_type, self.param_sampler, self.objective_function
        processing_pool = ProcessingPool(
            images_data = images_data,
            num_workers = num_workers,
            worker_function = regression_tree_worker,
            worker_params = worker_params)
        
        images_data = None # Free this copy
        try:
            tqdm.write(f'Training forest with {len(data_samples[0])} samples')
        
            self.training_lock = Lock()
            self.tqdm = tqdm(
                iterable = None,
                desc = 'Training tree  ',
                total = len(data_samples[0]) * self.max_depth,
                ascii = True
            )

            root_node = Node('0')
            self.add_node(root_node)
            work_data = (root_node.id, root_node.depth, *data_samples)
            processing_pool.enqueue_work_async(work_data)

            processing_pool.join_on_work_queue()
            self.is_trained = True
        except KeyboardInterrupt:
            tqdm.write(f'Stopping training due to KeyboardInterrupt')
        finally:
            processing_pool.finish()

class RegressionForest:
    def __init__(self,
        num_trees: int,
        max_depth: int,
        feature_type: FeatureType,
        param_sampler: Callable[[int], np.array],
        num_param_samples: int,
        objective_function: Callable[[np.array, np.array, np.array], float]):

        self.is_trained = False
        self.num_trees = num_trees
        self.num_param_samples = num_param_samples
        self.trees = [RegressionTree(
            max_depth = max_depth,
            feature_type = feature_type,
            param_sampler = param_sampler,
            num_param_samples = num_param_samples,
            objective_function = objective_function) for _ in range(num_trees)]

    def evaluate(self, samples: np.array, images_data: Tuple[np.array, np.array, np.array]):
        if not self.is_trained:
            raise Exception('Forest is not trained!')

        return [tree.evaulate(samples, images_data) for tree in self.trees]

    def train(self,
        data_dir: str,
        scene_name: str,
        train_image_indices: np.array,
        num_images_per_tree: int,
        num_samples_per_image: int,
        num_workers: int = cpu_count()):
        """
        Train this forest with the list of data samples given. The complete list of samples
        will be split among the self.num_trees trees and each tree trained on a subset of the
        samples.

        Parameters
        ----------
        data: DataHolder
            The data to train this forest with
        num_param_samples: int
            The number of parameters samples to train
        reset: bool = False
            Reset tree
        """
        loader = DataLoader(data_dir)

        for tree in tqdm(self.trees, ascii = True, desc = f'Training forest'):
            train_indices = np.random.choice(train_image_indices, size = num_images_per_tree, replace = False)
            tree.train(loader, scene_name, train_indices, num_samples_per_image, num_workers)

        self.is_trained = True
        self.train_image_indices = train_image_indices
        self.scene_name = scene_name
            