from time import sleep
from typing import Callable, Tuple
from tqdm import tqdm
import numpy as np
from multiprocessing import Process, cpu_count, shared_memory, JoinableQueue
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
    queue_work: JoinableQueue,
    queue_result: JoinableQueue,
    shm_meta: Tuple,
    worker_function: Callable[[any], Callable],
    worker_params: Tuple,
    worker_idx: int):
    images_data, close_shm = load_shared_memory(shm_meta)

    while True:
        try:
            done, work_data = queue_work.get()
            if (done == True):
                queue_work.task_done()
                break
            result = worker_function(images_data, work_data, worker_params)
            queue_result.put(result)
            queue_work.task_done()
        except Exception as e:
            print(f'Uncaught error in pool_worker!: {e}')
            break
        except KeyboardInterrupt:
            break
    close_shm()
    print(f'Stopping worker #{worker_idx}')

class ProcessingPool:
    def __init__(self, 
        images_data: Tuple[np.array, np.array, np.array],
        worker_function: Callable[[any], Callable] = None,
        worker_params: Tuple = (),
        num_workers: int = cpu_count()):
        self.shm_meta, self.unlink_shm = init_shared_memory(images_data)
        print(f'Initialized shared memory pool of {get_arrays_size_MB(images_data):3.3F}MB')

        self.queue_work = JoinableQueue()
        self.queue_result = JoinableQueue()
        self.worker_params = worker_params

        if not worker_function == None:
            self.workers = [Process(
                                target=pool_worker,
                                args=(self.queue_work, self.queue_result, self.shm_meta, worker_function, worker_params, i)
                            ) for i in range(num_workers)]
            
            for w in self.workers:
                w.start()
        else:
            self.workers = []

    def finish(self):
        try:
            # Signal to workers to finish
            for w in self.workers:
                self.queue_work.put((True, None))
            # Wait for workers to finish
            print(f'Waiting for workers to terminate')
            sleep(3)
            for w in self.workers:
                w.join(2)
                if not w.is_alive:
                    w.close()
                else:
                    w.terminate()
            print(f'All processing workers exited')
            self.unlink_shm()
            print(f'Shared memory pool freed')
            self.queue_work.close()
            self.queue_work.join_thread()
            self.queue_work.close()
            self.queue_work.join_thread()
        except Exception as e:
            print(f'Error stopping workers: ', e)

    def enqueue_work(self, work_data):
        self.queue_work.put((False, work_data))