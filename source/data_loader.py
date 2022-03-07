from operator import index
import os
from typing import Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image

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