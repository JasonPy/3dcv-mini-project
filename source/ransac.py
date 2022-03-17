from random import randint, random
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def initialize_hypotheses(image, forest, number_pixels, k, data_tuple):
    """
    Generate k hypotheses based on n (number_pixels) pixels, using the kabsch algorithm

    Args:
        image (ndarray): is used to retrieve the depth information
        number_pixels (int): the number of pixel that are used for the generation
        k (int): The number of hypotheses
    """
    shape = image.shape
    #random sampled pixels
    random_pixels = np.zeros((number_pixels, 3), dtype = np.uint16)
    #generated hypotheses
    hypotheses = []
    depths = np.zeros((number_pixels, ))

    for i in range(k):
        for j in range(number_pixels):
            random_x = np.random.randint(0, shape[0])
            random_y = np.random.randint(0, shape[1])
            
            depths[j] = image[random_x, random_y, 3]

            random_pixels[j,:] = 0, random_x, random_y

        #evaluate the forest with pixel (x,y)
        forest_modes = forest.evaluate(random_pixels, data_tuple)
        
        mode_matrix = sample_random_modes(forest_modes)
        
        #remove invalid modes/pixels
        mask = np.isfinite(mode_matrix)
        filtered_modes = np.reshape(mode_matrix[mask], (int(mode_matrix[mask].shape[0] / 3), 3))

        
        #check if at least 3 points are valid
        if filtered_modes.shape[0] < 3:
            print("##################### Resample")
            return initialize_hypotheses(image, forest, number_pixels, k, data_tuple)
        print("Finish")
        
        camera_space_cords = np.zeros(random_pixels.shape)
        camera_space_cords[:,:2] = random_pixels[:,1:]
        camera_space_cords[:,2] = depths
        
        filtered_pixels = np.reshape(camera_space_cords[mask], (int(camera_space_cords[mask].shape[0] / 3), 3))        
        
        transformation_matrix = get_transformation_matrix(filtered_pixels, filtered_modes)
        
        hypotheses.append(transformation_matrix)
        
    return hypotheses




def energy_function(predicted_positions, pixel, camera_hypothesis):
    """
    Calculate the energy for a given hypothesis based on a top hat error functino with a width defined by top_hat_error_width

    Args:
        predicted_position (_type_): predicted position
        pixel (_type_): 3d position, camera space
        camera_hypothesis (_type_): _description_
    """
    top_hat_error_width = 100
    print(camera_hypothesis.shape, pixel.shape)
    min_distance = np.min(np.linalg.norm(predicted_positions - (np.matmul(camera_hypothesis, pixel).T)))
    return int(~(min_distance > -1 * top_hat_error_width / 2 and min_distance < top_hat_error_width / 2))
 

def optimize(forest, image, k, data_tuple, number_pixels = 3, batch_size = 500):
    """
    Generate k hypothesis based on n (number_pixels) pixels and optimize the hypotheses

    Args:
        image (_type_): _description_
        k: the number of hypotheses
        number_pixels (int, optional): The number of pixels that are used to calculate the inital hypotheses. Defaults to 3.
        batch_size: The size of the batch used for the energy calculation.
        forest : The regression forest prediciting the modes.
    """
    
    #initialize k camera hypotheses
    hypotheses = initialize_hypotheses(image, forest, number_pixels, k, data_tuple)
    print('////////////////////////////////////////////////')

    
    energies = np.zeros(k)
        
    while len(hypotheses) > 1:
        #sample random set B of image coordinates
        pixel_batch = sample_pixel_batch(batch_size, image.shape)
        
        energies = np.zeros(len(hypotheses))        
        
        modes = forest.evaluate(pixel_batch, data_tuple)
        
        #remove invalid modes
        mask = np.isfinite(modes)
        modes = np.reshape(modes[mask], (int(modes[mask].shape[0] / 3), 3))
        pixel_batch = np.reshape(pixel_batch[mask], (int(pixel_batch[mask].shape[0] / 3), 3))
        
        pixel_inliers = {}
        for i in tqdm(range(len(hypotheses)), desc = "Calculating hypotheses energy"):
                pixel_inliers[i] = []
                for pixel_index in range(pixel_batch.shape[0]):
                    
                    depth = image[pixel_batch[pixel_index, 1], pixel_batch[pixel_index, 2], 3]
                    pixel = np.array([pixel_batch[pixel_index, 1], pixel_batch[pixel_index, 2], depth, 1]).T
                    
                    reshaped_modes = np.reshape(modes[pixel_index,:], (int(modes.shape[1] / 3), 3))
                    
                    reshaped_modes = np.append(reshaped_modes, np.ones((reshaped_modes.shape[0], 1)), axis = 1)
                    
                    energy = energy_function(reshaped_modes, pixel, hypotheses[i])
                    energies[i] += int(energy)

                    if energy == 0:
                        pixel_inliers[i].append(pixel_index)
                print(f'Energy:{energies}')
        print(energies)
        sorted_energy_index = np.argsort(energies)
        upper_half = int(len(sorted_energy_index) // 2)
        
        half_energie_index = sorted_energy_index[:upper_half]
        
        #remove half of the hypotheses with a high energy
        tmp = []
        for index in half_energie_index:            
            tmp.append(hypotheses[index])
        
        hypotheses = tmp

        #refine hypotheses based on inlieres and kabsch
        i = 0
        for hypothesis_index in sorted_energy_index[:upper_half]:
            pixels = pixel_batch[pixel_inliers[hypothesis_index],:]
            print("#########################################")
            print(pixels)
            print(len(pixel_inliers[hypothesis_index]))
            modes = forest.evaluate(pixels, data_tuple)
            
            depths = image[pixels[:, 1], pixels[:, 2], 3]

            camera_space_cords = np.zeros(pixels.shape)
            camera_space_cords[:,:2] = pixels[:,1:]
            camera_space_cords[:,2] = depths
            
            #sample random mode indices
            random_modes = sample_random_modes(modes)

            get_transformation_matrix(camera_space_cords, random_modes)
            hypotheses[i] = get_transformation_matrix(camera_space_cords, random_modes)
            i += 1
        
    
    return hypotheses[0], energies[0]
        
    

def sample_pixel_batch(batch_size, image_shape):
    x = np.linspace(0, image_shape[0] - 1, image_shape[0])
    y = np.linspace(0, image_shape[1] - 1, image_shape[1])
    pixel_batch = np.zeros((batch_size, 3), dtype = np.uint16)
    
    for i in range(batch_size):
        xx = np.random.randint(0, image_shape[0])
        yy = np.random.randint(0, image_shape[1])
        
        pixel_batch[i,:] = np.array([0, x[xx], y[yy]])
        
    return pixel_batch


def sample_random_modes(modes):
    upper_limit = modes.shape[1] / 3
    
    output = np.zeros((modes.shape[0], 3))
    #sample random mode 
    for i in range(modes.shape[0]):
        random_mode_index = np.random.randint(0, upper_limit) * 3
        mode = modes[i, random_mode_index:random_mode_index + 4]
        output[i,:] = mode
    
    return output

def get_transformation_matrix(a,b):
    """
    Computes a affine transformation matrix based on the kabsch algorithm

    Args:
        a (array): matrix a, containing points (row wise)
        b (array): matrix b, containing points (row wise)
    """
    #add homogenous coordinates
    ones = np.ones((a.shape[0], 1))
    a = np.append(a, ones, axis = 1)
    b = np.append(b, ones, axis = 1)
    
    #calculate centroid of the data   
    centroid_a = np.sum(a, axis = 0) / a.shape[0]
    centroid_b = np.sum(b, axis = 0) / b.shape[0]
    
    centered_a = a - centroid_a
    centered_b = b - centroid_b

    h = np.matmul(centered_a.T, centered_b)
    
    u, s, v = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.matmul(v, u.T)))
    dia = np.diag(np.array([1,1,1,d]))
    
    #get rotation matrix
    transformation_matrix = np.matmul(u, np.matmul(dia, v))
    
    #get translation part
    translation = centroid_b - np.matmul(transformation_matrix, centroid_a)
    
    transformation_matrix[:3, 3] = translation[:3].T

    return transformation_matrix