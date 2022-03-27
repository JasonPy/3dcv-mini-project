import numpy as np
from typing import Tuple
from utils import get_intrinsic_camera_matrix
from feature_extractor import INVALID_DEPTH_VALUE

class Ransac:
    
    def __init__(self, image_data: Tuple[np.array, np.array, np.array], forest, indices: np.array, number_pixles = 10, batch_size = 500, k = 1024, top_hat_width = 0.1):
        """_summary_

        Parameters
        ----------
        image_data (Tuple[np.array, np.array, np.array]): 
            The image data to evaluate
        forest : 
            forest for evaluation
        indices (np.array): 
            the indices in image_data to evaluate
        number_pixles (int, optional):  
            Defaults to 10.
        batch_size (int, optional): 
            Number of Pixels for hypotheses evaluation. Defaults to 500.
        k (int, optional): 
            Number of hypotheses. Defaults to 1024.
        top_hat_width (float, optional): 
            Allowed Error. Defaults to 0.1.
        """
        self.image_data = image_data
        self.forest = forest
        self.indices = indices
        self.inv_camera_matrix = np.linalg.inv(get_intrinsic_camera_matrix())
        self.number_pixels = number_pixles
        self.batch_size = batch_size
        self.k = k
        self.top_hat_width = top_hat_width
    
    
    def find_poses_parallel(self):
        pose = self.optimize(0)

        return (pose, self.indices[0])

    def find_poses(self):
        result = np.zeros(( self.indices.shape[0], 4, 4))
        for index in self.indices:
            pose = self.optimize(0)
            result[index,:,:] = pose            
        return result


    def optimize(self, index: int):
        """_summary_

        Parameters
        ----------
            index (int): Index of the image, for finding camera pose
        Returns
        ----------initia
            hypothesis (np.array): Estimated hypothesis
        """
        
        #initialize k camera hypotheses
        hypotheses = self.initialize_hypotheses(index)    

        #store for each pixel the modes    
        pixel_inliers = {}
        pixel_inliers_modes = {}
        
        valid_hypotheses_indices = np.arange(0, hypotheses.shape[2])

        #initialize for each hypothese a list
        for element in valid_hypotheses_indices:
            pixel_inliers[element] = []
            pixel_inliers_modes[element] = []

        energies = np.zeros(len(valid_hypotheses_indices))
        while len(valid_hypotheses_indices) > 1:
            #sample random set B of image coordinates
            pixel_batch = self.sample_pixel_batch(index).astype(np.int32)

            valid_depth_mask = ~((self.image_data[1][index, pixel_batch[:,1], pixel_batch[:,2]] == INVALID_DEPTH_VALUE) | (self.image_data[1][index, pixel_batch[:,1], pixel_batch[:,2]] == 0))
            pixel_batch = pixel_batch[valid_depth_mask,:]
            
            modes = self.forest.evaluate(pixel_batch, self.image_data)
            modes = np.asarray(modes)
            
            invalid = 0

            for pixel_index in range(pixel_batch.shape[0]):                    
                depth = self.image_data[1][index, pixel_batch[pixel_index, 1], pixel_batch[pixel_index, 2]]
                pixel = np.array([pixel_batch[pixel_index, 1], pixel_batch[pixel_index, 2], depth]).T
                
                pixel_modes = modes[:,pixel_index,:]
                #check if more than 0 modes are not none and remove them
                inv_c = ~np.isfinite(pixel_modes)
                if np.sum(~np.isfinite(pixel_modes)[:,0], axis = 0) < pixel_modes.shape[0]:
                    pixel_modes = pixel_modes[np.isfinite(pixel_modes)[:,0],:]
                #all modes are none skip pixel
                elif np.any(~np.isfinite(pixel_modes)):
                    invalid += 1
                    continue
                                
                pixel_modes = np.append(pixel_modes, np.ones((pixel_modes.shape[0], 1)), axis = 1)
                pixel_tmp = pixel
                pixel_tmp[2] = 1
                camera_coords = self.inv_camera_matrix @ pixel_tmp * depth / 1000
                
                camera_coords_h = np.append(camera_coords, 1)[np.newaxis, :, np.newaxis]
                camera_coords_h = np.repeat(camera_coords_h, hypotheses[:,:,valid_hypotheses_indices].shape[2], axis = 2)
                
                #matrix multiplication each column corresponds to one matrix multiplication
                world_coordinates = np.sum(hypotheses[:,:,valid_hypotheses_indices] * camera_coords_h, axis = 1)[:3,:].T
                world_coordinates = world_coordinates[:,:,np.newaxis]
                
                pixel_modes_tmp = np.swapaxes(pixel_modes[:,:3][np.newaxis, :,:],1, 2)
                distances = np.sqrt(np.sum((world_coordinates - pixel_modes_tmp)**2, axis = 1))
                mask = ~np.any(~(distances > self.top_hat_width), axis = 1)
                min_distance_indices = np.argmin(distances, axis = 1)
                min_distances = distances[:, min_distance_indices]
                outlier__mask= np.min(np.sqrt(np.sum((world_coordinates - pixel_modes_tmp)**2, axis = 1)), axis = 1) > self.top_hat_width
                test = np.all((mask == outlier__mask))                
                
                
                energies[valid_hypotheses_indices] += outlier__mask
                
                for counter, h_index in enumerate(valid_hypotheses_indices[~outlier__mask]):
                    pixel_inliers[h_index].append(np.array([index, pixel[0], pixel[1]]))
                    if np.any(~np.isfinite(pixel_modes[min_distance_indices[counter]])):
                         a = 1
                    pixel_inliers_modes[h_index].append(pixel_modes[min_distance_indices[counter], :3])

                    
            sorted_energy_indices = np.argsort(energies[valid_hypotheses_indices])       
            
            half_len = int(len(sorted_energy_indices) // 2)
            lower_half_indices = valid_hypotheses_indices[sorted_energy_indices[:half_len]]     
        
            #refine hypotheses based on inlieres and kabsch
            
            for hypothesis_index in lower_half_indices:
                pixels = np.asarray(pixel_inliers[hypothesis_index])
                if pixels.shape[0] == 0:
                    continue
                
                modes = pixel_inliers_modes[hypothesis_index]
                modes = np.asarray(modes)

                depths = self.image_data[1][index, pixels[:, 1], pixels[:, 2]]

                x_y_h = np.zeros(pixels.shape)
                x_y_h[:,:2] = pixels[:,1:]
                x_y_h[:,2] = np.ones((pixels.shape[0]))
                
                camera__scene_coord = (self.inv_camera_matrix @ x_y_h.T).T * depths[:,np.newaxis] / 1000
                                
                mask = np.isfinite(modes)
                mask = ~np.any(~mask, axis = 1)
                
                modes = modes[mask,:]
                camera__scene_coord = camera__scene_coord[mask,:]
                
                hypotheses[:,:,hypothesis_index] = self.get_transformation_matrix(camera__scene_coord, modes)           

            valid_hypotheses_indices = lower_half_indices
        return hypotheses[:,:,valid_hypotheses_indices[0]]
    
    def initialize_hypotheses(self, index: int):
        """
        Intialize k hypotheses based on pixel camera coordinates and their forest predictions 
        
        Parameters
        ----------
        index (int): 
            image index
        tqdm_pbar (_type_): 
            _description_

        Returns
        ----------
        np.array(4,4,k): 
            Hypotheses stored along axis = 2
        """
        shape = self.image_data[0][index].shape
            
        #generated hypotheses
        hypotheses = np.zeros((4,4, self.k))
        
        x = np.linspace(0, shape[0] - 1, shape[0])
        y = np.linspace(0, shape[1] - 1, shape[1])
        
        #switch x,y for method call since image is transposed
        xx, yy = np.meshgrid(x, y, indexing="ij")
                
        depths = self.image_data[1][index,:,:]
        valid_depths_mask = ~((depths == INVALID_DEPTH_VALUE) | (depths == 0))
        
        valid_x = xx[valid_depths_mask]
        valid_y = yy[valid_depths_mask]
        
        for step in range(self.k):
            forest_modes, random_pixels = self.get_random_pixel_modes(3, index, shape, valid_x, valid_y)
            
            random_pixels = random_pixels.astype(np.int32)
            depths = self.image_data[1][index][random_pixels[:,0], random_pixels[:,1]]

            x_y_h = np.hstack((random_pixels, np.ones(random_pixels.shape[0], dtype=np.int32)[:, np.newaxis]))
            
            # calculate for each pixel the camera scene coordinates
            for i in range(x_y_h.shape[0]):
                camera_coords = (self.inv_camera_matrix @ x_y_h[i,:].T)
                x_y_h[i,:] = camera_coords * depths[i] / 1000

            hypothesis = self.get_transformation_matrix(x_y_h[:,:3], forest_modes)
            
            hypotheses[:,:,step] = hypothesis

        return hypotheses



    def get_random_pixel_modes(self, n: int, index: int, shape, valid_xs: np.array, valid_ys: np.array):
        """
        Generate n random pixels and evaluate each pixel with the given forest.    

        Parameters
        ----------
            n (int): The number of random pixels
            index (int): image index
            shape (_type_): _description_
            image (_type_): _description_
            valid_xs (np.array): valid x values stored, as flatten array
            valid_ys (np.array): valid y values stored, as flatten array

        Returns
        ----------
            _type_: _description_
        """
        
    
        random_position = np.random.choice(valid_xs.shape[0], size= n, replace=False)
        pixels_xs = valid_xs[random_position]
        pixel_ys = valid_ys[random_position]
        
        random_pixels = np.hstack((np.repeat(index, n)[:,np.newaxis], pixels_xs[:, np.newaxis], pixel_ys[:,np.newaxis])).astype(np.int32)

        
        predictions = self.forest.evaluate(random_pixels, self.image_data)

        
        modes, pixel_mask = self.select_n_random_modes(predictions)
        
        if modes is None:
            return self.get_random_pixel_modes(n, index, shape, valid_xs, valid_ys)

        random_pixels = random_pixels[pixel_mask,:]
        modes = modes[:3, :]
        #omit index entry
        random_pixels = random_pixels[:3,1:]
        
        return modes, random_pixels          
        

    def sample_pixel_batch(self, index):
        
        x,y = self.get_valid_coordinates(index)
        random_position = np.random.choice(x.shape[0], size= self.batch_size, replace=False)

        pixels_xs = x[random_position]
        pixel_ys = y[random_position]
                    
        return np.stack((np.full(self.batch_size, index), pixels_xs, pixel_ys)).T
    

    def get_valid_coordinates(self, index):
        
        depths = self.image_data[1][index,:,:]
        shape = depths.shape
        
        x = np.linspace(0, shape[0] - 1, shape[0])
        y = np.linspace(0, shape[1] - 1, shape[1])
        
        #switch x,y for method call since image is transposed
        xx, yy = np.meshgrid(x, y, indexing="ij")      
  
        valid_depths_mask = ~((depths == INVALID_DEPTH_VALUE) | (depths == 0))
        
        valid_x = xx[valid_depths_mask]
        valid_y = yy[valid_depths_mask]
        
        return valid_x, valid_y
        
    def select_n_random_modes(self, modes, n = 3):
        """ Evalutes the modes. Checks if the modes are valid at least for n pixels. 

        Parameters
        ----------
        modes : predicted modes

        Returns
        ----------
        output : 
            the valid modes as array (valid_pixels, 3)
        mask : 
            boolean mask indicates which pixels have a valid mode
        """
        modes = np.asarray(modes)
        
        modes_mask = np.isfinite(modes[:, :, 0])
        
        #indicates the columns with valid modes
        valid_columns = np.any(modes_mask, axis = 0)
        
        #less than n pixel modes are valid
        if np.sum(valid_columns, axis = 0) < n:
            return None, None
                
        output = np.zeros((np.sum(valid_columns, axis = 0), 3))
    
        #iterate over valid_columns and sample a random tree mode
        for i, valid_column in enumerate(np.where(valid_columns == 1)[0]):
            #g
            indices = np.where(modes_mask[:,valid_column] == 1)[0]
            random_tree_index = np.random.choice(indices, 1)
            mode = modes[random_tree_index, valid_column, :]
            output[i,:] = mode
        
        return output, np.any(modes_mask, axis = 0)

    def sample_random_modes(self, modes):

        modes = np.asarray(modes)

        upper_limit = modes.shape[0]
        
        output = np.zeros((modes.shape[1], 3))
    
        #sample random mode 
        for i in range(modes.shape[1]):
            random_tree_index = np.random.randint(0, upper_limit)
            mode = modes[random_tree_index, i, :]
            output[i,:] = mode
        
        return output

    def get_transformation_matrix(self, a,b):
        """
        Computes an affine transformation matrix based on the kabsch algorithm

        Args:
            a (array): matrix a, containing points (row wise)
            b (array): matrix b, containing points (row wise)
        """

        #calculate centroid of the data 
        centroid_a = np.mean(a, axis = 0)
        centroid_b = np.mean(b, axis = 0)   

        centered_a = a - centroid_a
        centered_b = b - centroid_b

        h = np.matmul(centered_a.T, centered_b)
        
        u, s, v = np.linalg.svd(h)
        d = np.sign(np.linalg.det(np.matmul(v.T, u.T)))
        diag_matrix = np.diag(np.array([1,1,d]))
        
        #get rotation matrix
        transformation_matrix = np.matmul(v.T, np.matmul(diag_matrix, u.T))
        
        #get translation part
        translation = centroid_b - np.matmul(transformation_matrix, centroid_a)
        
        # transformation_matrix[:3, 3] = translation[:3].T
        output = np.zeros((4,4))
        output[:3,:3] = transformation_matrix
        output[:3, 3] = translation.T
        output[3,3] = 1
                
        return output