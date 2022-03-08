from typing import Callable, Tuple
from tqdm import tqdm
import numpy as np
from multiprocessing import Process, cpu_count, shared_memory, Queue
print = tqdm.write

from feature_extractor import generate_data_samples
from utils import get_arrays_size_MB

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

def pool_worker(
    queue_work: Queue,
    queue_result: Queue,
    shm_meta: Tuple,
    worker_function: Callable[[any], Callable],
    worker_params: Tuple):
    images_data, close_shm = load_shared_memory(shm_meta)

    while True:
        try:
            index, work_data = queue_work.get()
            if (index == -1):
                break
            result = worker_function(images_data, *work_data, *worker_params)
            queue_result.put((index, result))
        except Exception as e:
            print(f'Uncaught error in pool_worker!: {e}')
            break
        except KeyboardInterrupt:
            break
    close_shm()

class ProcessingPool:
    def __init__(self, 
        images_data: Tuple[np.array, np.array, np.array],
        worker_function: Callable[[any], Callable],
        worker_params: Tuple,
        num_workers: int = cpu_count()):
        self.shm_meta, self.close_shm = init_shared_memory(images_data)
        print(f'Initialized shared memory pool of {get_arrays_size_MB(images_data):3.3F}MB')

        self.queue_work = Queue()
        self.queue_result = Queue()
        self.worker_params = worker_params
        self.workers = [Process(
                            target=pool_worker,
                            args=(self.queue_work, self.queue_result, self.shm_meta, worker_function, worker_params)
                        ) for i in range(num_workers)]
        
        for w in self.workers:
            w.start()

    def stop_workers(self):
        for w in self.workers:
            self.queue_work.put((-1, None))
        for w in self.workers:
            w.join()
        print(f'All processing workers exited')
        self.close_shm()
        print(f'Shared memory pool freed')
        self.queue_work.close()
        self.queue_work.cancel_join_thread()

    def get_image_data(self):
        return load_shared_memory(self.shm_meta)

    def get_worker_params(self):
        return self.worker_params

    def process_work(self, work_datas):
        i_s = range(len(work_datas))
        for i in i_s:
            self.queue_work.put((i, work_datas[i]))

        results = [0] * len(i_s)
        for _ in tqdm(i_s, delay = 1, ascii = True, leave = False, desc = f'Training node  '):
            i, result = self.queue_result.get()
            results[i] = result
        return results