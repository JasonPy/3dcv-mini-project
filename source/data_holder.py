from typing import Union, Tuple
from tqdm import tqdm
import numpy as np
from multiprocessing import Process, cpu_count, shared_memory, Queue

from feature_extractor import FeatureExtractor, FeatureType

def init_shared_array(array: np.ndarray):
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    buffer_ndarray = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    buffer_ndarray[:] = array[:]
    return (shm.name, array.shape, array.dtype), shm

def load_shared_array(meta: Tuple) -> Tuple[np.ndarray, shared_memory.SharedMemory]:
    shm_name, shape, dtype = meta
    shm = shared_memory.SharedMemory(name=shm_name)
    return (np.ndarray(shape, dtype=dtype, buffer=shm.buf), shm)

def init_shared_memory(images_data: Tuple[np.array, np.array, np.array]):
    (images_rgb, images_depth, images_pose) = images_data
    shm_meta_rgb, shm_rgb = init_shared_array(images_rgb)
    shm_meta_depth, shm_depth = init_shared_array(images_depth)
    shm_meta_pose, shm_pose = init_shared_array(images_pose)

    def unlink():
        for shm in [shm_rgb, shm_depth, shm_pose]:
            shm.close()
            shm.unlink()

    return (shm_meta_rgb, shm_meta_depth, shm_meta_pose), unlink

def load_shared_memory(metadata):
    shm_meta_rgb, shm_meta_depth, shm_meta_pose = metadata
    images_rgb, shm_rgb = load_shared_array(shm_meta_rgb)
    images_depth, shm_depth = load_shared_array(shm_meta_depth)
    images_pose, shm_pose = load_shared_array(shm_meta_pose)

    def close():
        shm_rgb.close()
        shm_depth.close()
        shm_pose.close()
    
    images_data = (images_rgb, images_depth, images_pose)
    return images_data, close

def pool_worker(queue_work: Queue, queue_result: Queue, shm_meta: Tuple):
    images_data, close_shm = load_shared_memory(shm_meta)
    (images_rgb, images_depth, images_pose) = images_data
    feature_extractors = dict()

    while True:
        try:
            index, work_data = queue_work.get()
            if (index == -1):
                break
            (idx_s, p_s, param_sample, feature_type) = work_data
            output = np.ndarray((len(idx_s)), dtype=bool)

            for n in range(len(idx_s)):
                i = idx_s[n]
                if i not in feature_extractors:
                    feature_extractors[i] = FeatureExtractor(index = i, image_data = (images_rgb[i], images_depth[i], images_pose[i]))
                output[n] = feature_extractors[i].get_feature(p_s[n], param_sample, feature_type)

            queue_result.put((index, output))

        except Exception as e:
            break
    close_shm()

class ProcessingPool:
    def __init__(self, images_data: Tuple[np.array, np.array, np.array], num_workers: int = cpu_count()):
        self.shm_meta, self.close_shm = init_shared_memory(images_data)

        self.queue_work = Queue()
        self.queue_result = Queue()
        self.workers = [Process(
                            target=pool_worker,
                            args=(self.queue_work, self.queue_result, self.shm_meta)
                        ) for i in range(num_workers)]
        
        for w in self.workers:
            w.start()

    def stop_workers(self):
        for w in self.workers:
            self.queue_work.put((-1, None))
        for w in self.workers:
            w.join()
        self.close_shm()

    def get_features_for_param_sample(self, idx_s: np.array, p_s: np.array, param_samples: np.array, feature_type: FeatureType):
        i_s = range(len(param_samples))
        for i, param_sample in zip(i_s, param_samples):
            work_data = (idx_s, p_s, param_sample, feature_type)
            self.queue_work.put((i, work_data))

        results = [0] * len(i_s)
        for _ in tqdm(i_s, delay = 2, ascii = True, leave = False, desc = f'Training node  '):
            i, result = self.queue_result.get()
            results[i] = result
        return results

class DataHolder:
    def __init__(self,
        idx_s: np.array = np.ndarray((0), dtype=int),
        p_s: np.array = np.ndarray((0, 2), dtype=int),
        w_s: np.array = np.ndarray((0, 3), dtype=np.float64)):
        """
        A holder, which holds a list of data points sharing the same FeatureExtractor (image)
        
        We don't save a List[Sample] to more efficiently work on a collection of samples
        which share the same FeatureExtractor

        Parameters
        ----------
        idx_s: np.array
            Array of image indices associated with the samples 
        p_s: np.array
            Array of input features of the data points
        w_s: np.array
            Array of target responses of the data points
        """
        self.idx_s = idx_s
        self.p_s = p_s
        self.w_s = w_s

    def __len__(self):
        return len(self.p_s)

    # Yes, the type-hint for the DataHolder return-type has to be in quotes:
    # https://www.python.org/dev/peps/pep-0484/#forward-references
    def __getitem__(self, argument: Union[int, np.array]) -> Union[Tuple, 'DataHolder']:
        """
        Python magic function which implements the array access operator:
            "object.__getitem__(a) == object[a]"

        Supports accessing a single elemement (type(argument) == int) returning a Sample
        as well as masking the complete DataHolder with a mask array (type(argument) == np.array)

        Parameters
        ----------
        argument: int
            The index of the sample to return
        argument: np.array
            np.array describing a boolean mask for this sample-list

        Returns
        -------
        arg == int:
            A single Sample
        arg == np.array
            A DataHolder with containing only the samples on whose index the mask evaluates to true
        """
        if isinstance(argument, int):
            (self.idx_s[argument], self.p_s[argument], self.w_s[argument])
        
        else:
            if len(argument) != len(self.p_s):
                raise Exception(f'Mask of length {argument.shape[0]} cannot be applied to '
                                        f'DataHolder of size {len(self.p_s)}')
            idx_s_sliced = self.idx_s[argument]
            p_s_sliced = self.p_s[argument]
            w_s_sliced = self.w_s[argument]

            return DataHolder(idx_s_sliced, p_s_sliced, w_s_sliced)

    def add_samples(self, idx_s: np.array, p_s: np.array, w_s: np.array):
        self.idx_s = np.append(self.idx_s, idx_s, axis = 0)
        self.p_s = np.append(self.p_s, p_s, axis = 0)
        self.w_s = np.append(self.w_s, w_s, axis = 0)

    def is_empty(self):
        return len(self.idx_s) == 0

    def get_features_with_parameters(self, processing_pool: ProcessingPool, param_samples: np.array, feature_type: FeatureType) -> np.array:
        """
        Vectorized version of the Sample.get_feature_with_parameters
        """
        return processing_pool.get_features_for_param_sample(
            idx_s = self.idx_s,
            p_s = self.p_s,
            param_samples = param_samples,
            feature_type = feature_type)

    def get_all_sample_data_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return all sample data points (input feature and target response)
        in all SampleHolders.
        
        Returns
        -------
        (p_s, w_s) : Tuple[np.ndarray, np.ndarray]
            p_s: np.ndarray
                The input_feature data points (image coordinates)
            w_s: np.ndarray
                The target_response data points (3d world coordinates)
        """
        return (self.p_s, self.w_s)

def sample_from_data_set(images_data: Tuple[np.array, np.array, np.array], num_samples: int) -> DataHolder:
    """
    Samples the specified number of samples from each of the provided images.
    
    Parameters
    ----------
    images_data: Tuple[np.array, np.array, np.array]
        The image data
    num_samples: int
        The number of samples to draw from each image
    """
    data_holder = DataHolder()

    for i in range(images_data[0].shape[0]):
        image_data = (images_data[0][i], images_data[1][i], images_data[2][i])
        feature_extractor = FeatureExtractor(index = i, image_data = image_data)
        (p_s, w_s) = feature_extractor.generate_data_samples(num_samples)
        idx_s = np.full(p_s.shape[0], i, dtype=int)
        data_holder.add_samples(idx_s = idx_s, p_s = p_s, w_s = w_s)

    return data_holder
