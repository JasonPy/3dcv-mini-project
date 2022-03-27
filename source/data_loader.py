import os
import numpy as np

from typing import Tuple
from tqdm import tqdm
from PIL import Image

from feature_extractor import generate_data_samples

SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.metadata = dict()
        self.scenes = None

        # check which of the datasets is downloaded
        folders = [os.path.basename(folder) for folder in os.listdir(self.data_path)]
        self.scenes =  np.intersect1d(SCENES, folders)

        for scene in self.scenes:
            self.metadata[scene] = []
            scene_dir = os.path.join(self.data_path, scene)
            for segment_dir in os.listdir(scene_dir):
                segment_dir_path = os.path.join(scene_dir, segment_dir)
                if os.path.isdir(segment_dir_path):
                    files = sorted(os.listdir(segment_dir_path))
                    file_paths = [os.path.join(scene_dir, segment_dir, file) for file in files]
                    self.metadata[scene].extend(file_paths)
            self.metadata[scene] = sorted(self.metadata[scene])

    def get_dataset_length(self, scene_name: str):
        """
        Retrieve the length of a data set.
        """
        if scene_name in self.scenes:
            return len(self.metadata[scene_name]) // 3 # three types of images per sample
        else:
            raise Exception(f"The data set '{scene_name}' is not downloaded!")

    def load_dataset(self, scene_name: str, image_indices: np.array):
        """
        Convert one of the 7-scenes datasets into numpy arrays.
    
        Parameters
        ----------
        scene_name:
            the directory from where the data is obtained
        image_indices:
            the indices of the images to load

        Returns
        ----------
        dataset: Tuple[data_rgb: np.array,data_d: np.array,data_pose: np.array]
            Returns the images, respective depth maks and camera poses
        """

        if scene_name not in self.scenes:
            raise Exception(f"The data set '{scene_name}' is not downloaded!")
            
        files_to_load = []
        file_paths_for_scene = self.metadata[scene_name]
        for image_index in image_indices:
            for i in range(3):
                files_to_load.append(file_paths_for_scene[image_index * 3 + i])

        data_rgb = []
        data_d = []
        data_pose = []

        for file_path in tqdm(files_to_load, ascii = True, desc = f'Loading image data', dynamic_ncols = True, leave = False):
            if file_path.endswith(".color.png"):  
                image = np.array(Image.open(file_path))
                data_rgb.append(np.swapaxes(image, 0, 1))

            if file_path.endswith(".depth.png"):
                depth = np.array(Image.open(file_path))
                data_d.append(depth.T)
            
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
        total_samples = 0

        for i in tqdm(range(num_images), ascii = True, desc = 'Generating samples', dynamic_ncols = True, leave = False):
            (p_s, w_s) = generate_data_samples(images_data, i, num_samples)
            num_samples = len(p_s)
            total_samples += num_samples
            
            p_s_tot[i*num_samples:(i+1)*num_samples] = p_s
            w_s_tot[i*num_samples:(i+1)*num_samples] = w_s

        return p_s_tot[0:total_samples], w_s_tot[0:total_samples]

