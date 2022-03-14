import os
from typing import Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image
from feature_extractor import generate_data_samples

SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.metadata = dict()

        for scene in SCENES:
            self.metadata[scene] = []
            scene_dir = os.path.join(self.data_path, scene)

            for segment_dir in os.listdir(scene_dir):
                segment_dir_path = os.path.join(scene_dir, segment_dir)
                if os.path.isdir(segment_dir_path):
                    files = sorted(os.listdir(segment_dir_path))
                    file_paths = [os.path.join(scene_dir, segment_dir, file) for file in files]
                    self.metadata[scene].extend(file_paths)

    def get_dataset_length(self, scene_name: str):
        return len(self.metadata[scene_name]) // 3 # three types of images per sample

    def load_dataset(self, scene_name: str, index_slice: Tuple[int]):
        """
        Convert one of the 7-scenes datasets into numpy arrays.
    
        Parameters
        ----------
        scene_name:
            the directory from where the data is obtained
        index_slice:
            the indices of the images to load

        Returns
        ----------
        dataset: Tuple[
            data_rgb: List,
            data_d: List,
            data_pose: List
        ]"""
        i_low = index_slice[0] * 3
        i_high = index_slice[1] * 3
        files_to_load = self.metadata[scene_name][i_low:i_high]

        data_rgb = []
        data_d = []
        data_pose = []

        for file_path in tqdm(files_to_load, ascii = True, desc = f'Loading image data'):
            if file_path.endswith(".color.png"):  
                image = np.array(Image.open(file_path))
                data_rgb.append(image)

            if file_path.endswith(".depth.png"):
                depth = np.array(Image.open(file_path))
                data_d.append(depth)
            
            if file_path.endswith(".pose.txt"):
                pose = np.loadtxt(file_path)
                data_pose.append(pose)

        return np.array(data_rgb), np.array(data_d), np.array(data_pose)

    def sample_from_data_set(self, images_data: Tuple[np.array, np.array, np.array], num_samples: int) -> Tuple[np.array, np.array]:
        """
        Samples the specified number of samples from each of the provided images.
        
        Parameters
        ----------
        images_data: Tuple[np.array, np.array, np.array]
            The image data
        num_samples: int
            The number of samples to draw from each image
        """
        num_images = images_data[0].shape[0]
        p_s_tot = np.zeros((num_samples * num_images, 3), dtype=np.int16)
        w_s_tot = np.zeros((num_samples * num_images, 3), dtype=np.float32)

        for i in tqdm(range(num_images), ascii = True, desc = 'Generating samples'):
            (p_s, w_s) = generate_data_samples(images_data, i, num_samples)
            
            p_s_tot[i*num_samples:(i+1)*num_samples] = p_s
            w_s_tot[i*num_samples:(i+1)*num_samples] = w_s

        return p_s_tot, w_s_tot