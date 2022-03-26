import numpy as np

from typing import Callable
from multiprocessing import cpu_count
from typing import Tuple
from numba import njit, jit, prange
from numpy.random import choice, uniform
from tqdm import tqdm

from feature_extractor import FeatureType, get_features_for_samples
from processing_pool import ProcessingPool
from utils import get_mode, vector_3d_array_variance, split_set, millis
from data_loader import DataLoader

@njit
def param_sampler(num_samples: int) -> np.array:
    """
    Generate parameter samples according to (2.2) and (4.4)
    for each node.
    """
    rgb_coords = np.array([0, 1, 2])
    tau = uniform(-100, 100, num_samples)
    delta1x = uniform(-130, 130, num_samples)
    delta1y = uniform(-130, 130, num_samples)
    delta2x = uniform(-130, 130, num_samples)
    delta2y = uniform(-130, 130, num_samples)
    c1 = choice(rgb_coords, num_samples, replace=True)
    c2 = choice(rgb_coords, num_samples, replace=True)
    return np.stack((tau, delta1x, delta1y, delta2x, delta2y, c1, c2)).T.astype(np.float64)

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
        # if no samples present, bad split
        return np.inf
    
    # get fraction of samples in left, right split
    frac_left = len(w_left) / num_samples
    frac_right = len(w_right) / num_samples

    # get variance of left, right split
    var_left = 0 if frac_left == 0 else vector_3d_array_variance(w_left)
    var_right = 0 if frac_right == 0 else vector_3d_array_variance(w_right)

    return vector_3d_array_variance(w_complete) - (frac_left * var_left + frac_right * var_right)

@jit(nopython = True, parallel = True)
def calculate_scores_for_params(image_data, p_s, w_s, param_samples, objective_function, feature_type):
    """
    Calculate the scores according to the best split with respect to
    the parameter samples. 
    PARAMETERS
    ----------
    image_data: np.array
    p_s
    w_s
    param_samples
    objective_function
    feature_type
    """
    num_samples = len(param_samples)
    scores = np.full(num_samples, 0, dtype=np.float64)
    for i in prange(num_samples):
        mask_valid, mask_split = get_features_for_samples(image_data, p_s, param_samples[i], feature_type)
        w_s_valid = w_s[mask_valid]
        set_left, set_right = split_set(w_s_valid, mask_split)
        
        #check if split is invalid if so set energy to inf
        if set_left.shape[0] == 0 or set_right.shape[0] == 0:
            scores[i] = np.inf
        else:
            scores[i] = objective_function(w_complete = w_s, w_left = set_left, w_right = set_right)
    return scores

class TreeWorkerResult:
    def __init__(self,
        node_id: str,
        is_leaf: bool,
        progress: int = 0,
        response: np.array = None,
        params: np.array = None,
        set_left: Tuple[np.array, np.array] = None,
        set_right: Tuple[np.array, np.array] = None,
        lengths: Tuple[int, int, int, int] = None,
        timings: Tuple[float, float] = None):
        self.node_id = node_id
        self.is_leaf = is_leaf
        self.progress = progress
        self.response = response
        self.params = params
        self.set_left = set_left
        self.set_right = set_right
        self.lengths = lengths
        self.timings = timings

def regression_tree_worker(image_data, work_data, worker_params):
    # Extract work data
    (tree) = worker_params
    node_id, depth, p_s, w_s = work_data

    # Auxiliary local variables
    tree_levels_below = tree.max_depth - depth
    len_data = len(p_s)
    is_leaf_node = False
    progress = 0
    # _ms_start = millis()

    # Check if this node should not be trained, make it leaf node
    if len_data == 1 or depth == tree.max_depth:
        is_leaf_node = True
        progress += len_data
        response = get_mode(w_s)
        result = TreeWorkerResult(node_id = node_id, is_leaf =  True, response = response)

    if not is_leaf_node:
        # Generate parameter samples
        param_samples = param_sampler(tree.num_param_samples)
        scores = calculate_scores_for_params(
            image_data = image_data,
            p_s = p_s,
            w_s = w_s,
            param_samples = param_samples,
            objective_function = tree.objective_function,
            feature_type = tree.feature_type)

        # _delta_get_features = millis() - _ms_start
        
        # Find best parameter and calculate split (again, I know)
        max_score_index = np.argmin(scores)
        best_params = param_samples[max_score_index]
        mask_valid, mask_split = get_features_for_samples(image_data, p_s, best_params, tree.feature_type)

        # Split the input data        
        w_s_valid = w_s[mask_valid]
        w_s_left, w_s_right = split_set(w_s_valid, mask_split)
        p_s_valid = p_s[mask_valid]
        p_s_left, p_s_right = split_set(p_s_valid, mask_split)

        # Calculate lengths
        len_invalid = np.sum(~mask_valid)
        len_left = np.sum(mask_split)
        len_right = np.sum(~mask_split)

        # Report training progress
        # _delta_split = millis() - _delta_get_features - _ms_start
        progress += len_data

        if len_invalid == len_data:
            # All samples are invalid. Edge case; not observed so far
            print(f'Error in node {node_id}: All samples considered invalid')
            result = TreeWorkerResult(node_id = node_id, is_leaf = True, response = w_s[0])

        elif len_right == 0 or len_left == 0:
            # All samples are split to one side -> this should be a leaf node
            splitr_len = 0
            splitl_len = 0

            while splitr_len == 0 or splitl_len == 0:

                param_samples = param_sampler(tree.num_param_samples)
                scores = calculate_scores_for_params(
                    image_data = image_data,
                    p_s = p_s,
                    w_s = w_s,
                    param_samples = param_samples,
                    objective_function = tree.objective_function,
                    feature_type = tree.feature_type)

                # _delta_get_features = millis() - _ms_start
                
                # Find best parameter and calculate split (again, I know)
                max_score_index = np.argmin(scores)
                best_params = param_samples[max_score_index]
                mask_valid, mask_split = get_features_for_samples(image_data, p_s, best_params, tree.feature_type)

                splitl_len = np.sum(mask_split)
                splitr_len = np.sum(~mask_split)

                # Split the input data        
                w_s_valid = w_s[mask_valid]
                w_s_left, w_s_right = split_set(w_s_valid, mask_split)
                p_s_valid = p_s[mask_valid]
                p_s_left, p_s_right = split_set(p_s_valid, mask_split)
                if splitr_len != 0 and splitl_len != 0:
                    len_invalid = np.sum(~mask_valid)
                    progress += len_invalid * tree_levels_below

                    result = TreeWorkerResult(
                        node_id = node_id,
                        is_leaf = False,
                        params = best_params,
                        set_left = (p_s_left, w_s_left),
                        set_right = (p_s_right, w_s_right),
                        lengths = (len_data, len_invalid, len_left, len_right))                

        else:
            # Report training progress on invalid nodes, trigger next node training
            progress += len_invalid * tree_levels_below # invalid are considered "done" for all levels below
            result = TreeWorkerResult(
                node_id = node_id,
                is_leaf = False,
                params = best_params,
                set_left = (p_s_left, w_s_left),
                set_right = (p_s_right, w_s_right),
                lengths = (len_data, len_invalid, len_left, len_right))

    if is_leaf_node:
        # Report training progress ("skipped" calculations since this is a leaf node)
        progress += len_data * tree_levels_below 

    result.progress = progress
    return result

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
        self.depth = depth
        self.params = np.array([0], dtype=np.float64)
        self.response = np.array([.0, .0, .0])
        self.node_id_left = ''
        self.node_id_right = ''

    def is_leaf(self):
        return (self.node_id_left == '') and (self.node_id_right == '')

    def evaluate(self,
        images_data: Tuple[np.array, np.array, np.array],
        samples: np.array,
        tree: 'RegressionTree'):
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
            return np.full((len(samples), self.response.shape[0]), self.response)
        else:
            outputs = np.full((len(samples), 3), -np.inf)
            mask_valid, mask_split = get_features_for_samples(images_data, samples, self.params, tree.feature_type)
            samples_valid, _ = split_set(samples, mask_valid)
            split_left, split_right = split_set(samples_valid, mask_split)
            
            left_child = tree.nodes[self.node_id_left]
            right_child = tree.nodes[self.node_id_right]

            response_left = left_child.evaluate(images_data, split_left, tree)
            response_right = right_child.evaluate(images_data, split_right, tree)
            outputs_masked = outputs[mask_valid].copy()
            outputs_masked[mask_split] = response_left
            outputs_masked[~mask_split] = response_right
            outputs[mask_valid] = outputs_masked
            return outputs

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

    def handle_node_result(self, result: TreeWorkerResult):        
        if not result.node_id in self.nodes:
            raise Exception(f'Could not find node with id {result.node_id}')

        self.progress += result.progress
        self.tqdm.update(result.progress)

        node = self.nodes[result.node_id]
        if result.is_leaf:
            node.response = result.response
        else:
            # Create new nodes
            node_left = Node(node.id + '0', node.depth + 1)
            node_right = Node(node.id + '1', node.depth + 1)
            self.add_node(node_left)
            self.add_node(node_right)

            # Save this node's training result
            node.params = result.params
            node.node_id_left = node_left.id
            node.node_id_right = node_right.id
            
            # Enqueue the new nodes for training
            p_s_left, w_s_left = result.set_left
            p_s_right, w_s_right = result.set_right
            train_left_work_data = (node_left.id, node_left.depth, p_s_left, w_s_left)
            train_right_work_data = (node_right.id, node_right.depth, p_s_right, w_s_right)
            self.processing_pool.enqueue_work(train_left_work_data)
            self.processing_pool.enqueue_work(train_right_work_data)

            # len_data, len_invalid, len_left, len_right = result.lengths
            # _delta_get_features, _delta_split = result.timings
            # _str_split = f'| {len_data:10} in | {len_invalid:8} inval | {len_left:8} left | {len_right:8} right | {_delta_split:4.0F}ms split |'
            # _str_features = f'{len_data * self.num_param_samples:13} samples | {_delta_get_features:8.0F}ms eval |'
            # kilo_it_per_sec_str = f'{(len_data * self.num_param_samples) / (_delta_get_features + _delta_split):.1F}'
            # tqdm.write(f'Node trained         {_str_split} {_str_features} {kilo_it_per_sec_str:7}Kit/s | {node.id:16} id |')

        # I don't really know if this is necessary. I want pointers :(
        self.nodes[result.node_id] = node
            
    def evaulate(self, samples: np.array, images_data: Tuple[np.array, np.array, np.array]):
        if not self.is_trained:
            raise Exception('Error: Tree is not trained yet!')

        results = self.nodes['0'].evaluate(
            images_data = images_data,
            samples = samples, 
            tree = self)
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

        tqdm.write(f'Training forest with {len(data_samples[0]):.2E} samples') # images * samples_per_img
        self.total_iterations = len(data_samples[0]) * (self.max_depth + 1)
        self.progress = 0

        with tqdm(
            iterable = None,
            desc = 'Training tree  ',
            smoothing = 0.05,
            dynamic_ncols = True,
            unit_scale = True,
            mininterval = 0.2,
            maxinterval = 2,
            total = self.total_iterations,
            ascii = True) as tqdm_progress:

            self.tqdm = tqdm_progress
            self.processing_pool = ProcessingPool(
                images_data = images_data,
                num_workers = num_workers,
                worker_function = regression_tree_worker,
                worker_params = (self))
            images_data = None # Free this copy
            queue_work = self.processing_pool.queue_work
            queue_result = self.processing_pool.queue_result

            was_interruped = False

            try:
                root_node = Node('0')
                self.add_node(root_node)
                work_data = (root_node.id, root_node.depth, data_samples[0], data_samples[1])
                self.processing_pool.enqueue_work(work_data)

                while not (self.progress == self.total_iterations):
                    self.handle_node_result(queue_result.get())
                    queue_result.task_done()

                tqdm.write('Training complete.')
                queue_work.join()
                tqdm.write('Work queue emptied.')
                self.is_trained = True

            except KeyboardInterrupt:
                tqdm.write(f'Stopping training due to KeyboardInterrupt')
                was_interruped = True

            finally:
                self.processing_pool.finish()

                # Cleanup class attributes not to be serialized
                self.processing_pool = None
                self.tqdm = None
                if was_interruped:
                    raise KeyboardInterrupt()

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
            max_depth = max_depth - 1,
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

        try:
            with tqdm(self.trees, ascii = True, desc = f'Training forest', dynamic_ncols = True) as trees:
                for tree in trees:
                    train_indices = np.random.choice(train_image_indices, size = num_images_per_tree, replace = False)
                    np.save(f"train_indices", train_indices,allow_pickle=True, fix_imports=True)                    
                    tree.train(loader, scene_name, train_indices, num_samples_per_image, num_workers)

                self.is_trained = True
                self.train_image_indices = train_image_indices
                self.scene_name = scene_name
        except KeyboardInterrupt:
            pass            