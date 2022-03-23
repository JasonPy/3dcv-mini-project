from distutils.log import error
import faulthandler
from operator import inv
from random import randint, random
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import faulthandler
import open3d

def initialize_hypotheses(image, forest, number_pixels, k, data_tuple, index, inv_camera_matrix, tqdm_pbar):
    """
    Generate k hypotheses based on n (number_pixels) pixels, using the kabsch algorithm

    Args:
        image (ndarray): is used to retrieve the depth information
        number_pixels (int): the number of pixel that are used for the generation
        k (int): The number of hypotheses
        index : the position of the image in the data tuple
    """
    shape = image.shape
        
    #generated hypotheses
    hypotheses = []
    
    

    for i in range(k):
        # transformation_matrix = get_initial_transformation(number_pixels, index, forest, data_tuple, shape, image, inv_camera_matrix)
        forest_modes, pixels = get_random_pixel_modes(number_pixels, index, forest, data_tuple, shape, image)       
        
        # print(pixels)
        test = np.zeros_like(pixels, dtype=np.float64)
        test[:,:2] = pixels[:,:2] #* pixels[:,2][:,np.newaxis]
        test[:,2] = np.ones(test.shape[0])
        # print(test)
        # print(test)
        
        ones = np.ones((test.shape[0], 1))
        # test = np.append(test, ones, axis = 1)
        
        for i in range(test.shape[0]):
            camera_coords = (inv_camera_matrix @ test[i,:].T)
            test[i,:] = camera_coords * pixels[i, 2] / 1000
        
        #transformation  mean <0.05 ? else resample
        transformation_matrix = get_transformation_matrix(test[:,:3], forest_modes)
 
            
        ones = np.ones((forest_modes.shape[0], 1))

        forest_modes = np.append(forest_modes, ones, axis = 1)
        test = np.append(test, ones, axis = 1)
        # print(forest_modes)
        # print((transformation_matrix @ test[0,:].T)[:,np.newaxis].T)
        # print(np.linalg.norm(forest_modes - (transformation_matrix @ test[0,:].T)[:,np.newaxis].T, axis = 1))
        
        hypotheses.append(transformation_matrix)
        tqdm_pbar.update(1)

    return hypotheses



def get_random_pixel_modes(n, index, forest, data_tuple, shape, image):
    """
    Generate n random pixels and evaluate each pixel with the given forest.
    If the number of valid modes is smaller than 3 recursivly call this method again.
    
    Args:
        n (_type_): _description_
        index (_type_) : the index of the image in the data_tuple
        forest (_type_): _description_
        data_tuple (_type_): _description_
        shape (_type_): _description_
        image (_type_): _description_

    Returns:
        _type_: Random selected modes, random_pixels
    """
    
    random_pixels = np.zeros((n, 3), dtype = np.int16)
    depths = np.zeros((n, ))
    x = np.random.choice(shape[0], size= n, replace=True)
    y = np.random.choice(shape[1], size= n, replace=True)

    depths = image[x, y, 3]
    if np.any(depths == 65535) or np.any(depths == 0):
        return get_random_pixel_modes(n, index, forest, data_tuple, shape, image)
    
    indices = np.repeat(index, x.shape[0])
    random_pixels = np.hstack((indices[:,np.newaxis],x[:,np.newaxis], y[:,np.newaxis]))

    #       )
    # for j in range(n):
    #         random_x = np.random.randint(0, shape[0])
    #         random_y = np.random.randint(0, shape[1])
            
    #         depths[j] = image[random_x, random_y, 3]
    #         if depths[j] == 65535:
    #             print("invalid depth")
    #             return get_random_pixel_modes(n, index, forest, data_tuple, shape, image)
            
    #         random_pixels[j,:] = index, random_x, random_y

    #evaluate the forest with pixel (x,y)
    predictions = forest.evaluate(random_pixels, data_tuple)

    # get random mode (3)
    mode_matrix = sample_random_modes(predictions)
    
    #remove invalid modes/pixels
    mask = np.isfinite(mode_matrix)
    mask = ~np.any(~mask, axis = 1)

    filtered_modes = mode_matrix[mask,:]

    
    #check if at least 3 points are valid
    if filtered_modes.shape[0] < 3:
        return get_random_pixel_modes(n, index, forest, data_tuple, shape, image)
    
    x_y_d = np.zeros(random_pixels.shape)
    x_y_d[:,:2] = random_pixels[:,1:]
    x_y_d[:,2] = depths

    x_y_d = x_y_d[mask,:]
        
    return filtered_modes, x_y_d


def energy_function(predicted_positions, pixel, inv_camera_matrix, camera_hypothesis):
    """
    Calculate the energy for a given hypothesis based on a top hat error functino with a width defined by top_hat_error_width

    Args:
        predicted_position (_type_): predicted position
        pixel (_type_): 3d position, camera space
        camera_hypothesis (_type_): _description_
        inv_camera_matrix: the inverse camera matrix
    """
    
    tmp_pixel = np.zeros_like(pixel, dtype=np.float64)
    tmp_pixel[:2] = pixel[:2] 
    tmp_pixel[2] = 1
    # tmp_pixel[:3] = tmp_pixel[:3] / 1000.0
    t = inv_camera_matrix @ tmp_pixel
    t= t * pixel[2] / 1000
    t = np.append(t,1)
    # print(tmp_pixel)
    # print(f"Camera \n {camera_hypothesis}")
    # print(f" inv camera \n {inv_camera_matrix}")
    # print(np.matmul(np.linalg.inv(camera_hypothesis), inv_camera_matrix @ tmp_pixel).T)
    
    top_hat_error_width = 2
    min_distance = np.min(np.linalg.norm(predicted_positions[:,:3] - ((camera_hypothesis @ t))[:3]))
    
    # print(predicted_positions - (np.matmul(camera_hypothesis, inv_camera_matrix @ tmp_pixel).T))
    
    return min_distance
 

def optimize(forest, image, k, index, data_tuple, inv_camera_matrix, number_pixels = 3, batch_size = 500):
    """
    Generate k hypothesis based on n (number_pixels) pixels and optimize the hypotheses

    Args:
        forest : The regression forest prediciting the modes.
        image (_type_): _description_
        k: the number of hypotheses
        index : the position of the image in the data_tuple
        data_tuple : Tuple[array, array, array] the rgb, depth image and camera pose
        inv_camera_matrix : the inverse of the camera matrix
        number_pixels (int, optional): The number of pixels that are used to calculate the inital hypotheses. Defaults to 3.
        batch_size: The size of the batch used for the energy calculation.
    """
    pbar = tqdm(total = k, desc="Intialize hypotheses")
    
    #initialize k camera hypotheses

    hypotheses = initialize_hypotheses(image, forest, number_pixels, k, data_tuple, index, inv_camera_matrix, pbar)
    pbar.clear()
    faulthandler.enable()
    pixel_inliers = {}
    errors = {}
    
    valid_hypotheses = len(hypotheses)
    valid_hypotheses_indices = np.arange(0, len(hypotheses))
    
    #initialize for each hypothese a list
    for element in valid_hypotheses_indices:
          pixel_inliers[element] = []

    energies = np.zeros(len(valid_hypotheses_indices))
    total_points = 0
    
    while len(valid_hypotheses_indices) > 1:
        #sample random set B of image coordinates
        pixel_batch = sample_pixel_batch(batch_size, image.shape, index, image[:,:,3])
        modes = forest.evaluate(pixel_batch, data_tuple)
        modes = np.asarray(modes)
        
        invalid = 0
        t1 = time.time()
        for pixel_index in range(pixel_batch.shape[0]):                    
            depth = image[pixel_batch[pixel_index, 1], pixel_batch[pixel_index, 2], 3]
            pixel = np.array([pixel_batch[pixel_index, 1], pixel_batch[pixel_index, 2], depth]).T
            
            pixel_modes = modes[:,pixel_index,:]
            
            #check if more than 0 modes are not none and remove them
            inv_c = ~np.isfinite(pixel_modes)
            if np.sum(~np.isfinite(pixel_modes)[:,0], axis = 0) < pixel_modes.shape[0]:
                pixel_modes = pixel_modes[np.isfinite(pixel_modes)[:,0],:]
            #all modes are none skip pixel
            elif np.any(~np.isfinite(pixel_modes)):

                continue
            if depth == 65535:

                continue
            
            pixel_modes = np.append(pixel_modes, np.ones((pixel_modes.shape[0], 1)), axis = 1)

            for i,valid_index in enumerate(valid_hypotheses_indices):
                energy = energy_function(pixel_modes, pixel, inv_camera_matrix, hypotheses[valid_index])
                energies[valid_index] += int(energy>0.1)

                if energy <= 0.1:
                    pixel_inliers[valid_index].append(np.array([index, pixel[0], pixel[1]]))
        # #TODO: change of the order of the loops
        # for i,valid_index in enumerate(valid_hypotheses_indices): #tqdm(range(valid_hypotheses), desc = "Calculate energy"):
                
        #         errors[valid_index] = 0
        #         hypothesis = hypotheses[valid_index]
                
        #         for pixel_index in range(pixel_batch.shape[0]):                    
        #             depth = image[pixel_batch[pixel_index, 1], pixel_batch[pixel_index, 2], 3]
        #             pixel = np.array([pixel_batch[pixel_index, 1], pixel_batch[pixel_index, 2], depth]).T
                    
        #             pixel_modes = modes[:,pixel_index,:]
                    
        #             #check if more than 0 modes are not none and remove them
        #             inv_c = ~np.isfinite(pixel_modes)
        #             if np.sum(~np.isfinite(pixel_modes)[:,0], axis = 0) < pixel_modes.shape[0]:
        #                 pixel_modes = pixel_modes[np.isfinite(pixel_modes)[:,0],:]
        #             #all modes are none skip pixel
        #             elif np.any(~np.isfinite(pixel_modes)):
        #                 if i == 0:
        #                     invalid += 1
        #                 continue
        #             if depth == 65535:
        #                 if i == 0:
        #                     invalid += 1
        #                 continue
                    
        #             pixel_modes = np.append(pixel_modes, np.ones((pixel_modes.shape[0], 1)), axis = 1)
                    
        #             energy = energy_function(pixel_modes, pixel, inv_camera_matrix, hypothesis)
        #             errors[i] += energy
        #             energies[valid_index] += int(energy>0.1)

        #             if energy <= 0.1:
        #                 pixel_inliers[valid_index].append(np.array([index, pixel[0], pixel[1]]))
        print(time.time() - t1)
        total_points += (batch_size - invalid)
                
        sorted_energy_indices = np.argsort(energies[valid_hypotheses_indices])       
        
        half_len = int(len(sorted_energy_indices) // 2)
        lower_half_indices = valid_hypotheses_indices[sorted_energy_indices[:half_len]]
        
        #check if hypothesis has inlier
        if len(valid_hypotheses_indices) == len(energies):
            mask = energies[lower_half_indices] < batch_size - invalid
            lower_half_indices = lower_half_indices[mask]
        
        #remove half of the hypotheses with a high energy
        # tmp = []
        # for i in lower_half_indices:            
        #     tmp.append(hypotheses[i])
        
        # hypotheses = tmp
        
       
        #refine hypotheses based on inlieres and kabsch
        i = 0
        
        pbar = tqdm(total = lower_half_indices.shape[0], desc = "optimize hypotheses")
        t1 = time.time()
        for hypothesis_index in lower_half_indices:
            pixels = np.asarray(pixel_inliers[hypothesis_index])
            if pixels.shape[0] == 0:
                # pbar.update(1)
                continue
            
            modes = forest.evaluate(pixels, data_tuple)
            
            depths = image[pixels[:, 1], pixels[:, 2], 3]

            x_y_h = np.zeros(pixels.shape)
            x_y_h[:,:2] = pixels[:,1:]
            x_y_h[:,2] = np.ones((pixels.shape[0]))
            
            camera__scene_coord = (inv_camera_matrix @ x_y_h.T).T * depths[:,np.newaxis] / 1000
            
            #sample random mode indices
            random_modes = sample_random_modes(modes)
            
            mask = np.isfinite(random_modes)
            mask = ~np.any(~mask, axis = 1)
            
            random_modes = random_modes[mask,:]
            x_y_h = x_y_h[mask,:]
            
            hypotheses[hypothesis_index] = get_transformation_matrix(camera__scene_coord, random_modes)           

            pbar.update(1)
        pbar.close()
        print(f'optimize time {time.time() - t1}')

        valid_hypotheses_indices = lower_half_indices
    return hypotheses[valid_hypotheses_indices[0]], energies[valid_hypotheses_indices[0]]
        
    

def sample_pixel_batch(batch_size, image_shape, index, depth):

    coordinate_range = np.arange(image_shape[0] * image_shape[1], dtype=np.int32)
    i = 0
    p_s = np.zeros((batch_size, 3), dtype=np.int32)
    while i < batch_size:
        x_y_s = np.random.choice(coordinate_range, 1, replace=False)
        x_s = x_y_s // 480
        y_s = x_y_s % 480
        
        if depth[x_s, y_s] != 65535 and depth[x_s, y_s] != 0:
            p_s[i,:] = index, x_s, y_s
            i += 1
    return p_s


def sample_random_modes(modes):

    modes = np.asarray(modes)

    upper_limit = modes.shape[0]
    
    output = np.zeros((modes.shape[1], 3))
   
    #sample random mode 
    for i in range(modes.shape[1]):
        random_tree_index = np.random.randint(0, upper_limit)
        mode = modes[random_tree_index, i, :]
        output[i,:] = mode
    
    return output

def get_transformation_matrix(a,b):
    """
    Computes an affine transformation matrix based on the kabsch algorithm

    Args:
        a (array): matrix a, containing points (row wise)
        b (array): matrix b, containing points (row wise)
    """
    #add homogenous coordinates
    ones = np.ones((a.shape[0], 1))
    # a = np.append(a, ones, axis = 1)
    # b = np.append(b, ones, axis = 1)

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
    
    q = (output @ np.append(a, ones, axis = 1).T).T
    mean_distance = np.mean(np.linalg.norm(b - q[:, :3], axis = 1))
    # print(f'Distance: \n {np.linalg.norm(b - q[:, :3], axis = 1)}')
    # points = np.vstack((b,q[:,:3],a))
    
    # colors = np.zeros((9,3))
    # # colors[3:6,:] = np.repeat(np.array([[1, 0, 0]],),3, axis = 0)
    # colors[0,:] = 0, 1, 0
    # colors[1,:] = 1, 0, 0
    # colors[2,:] = 0, 0, 1
    
    # colors[3,:] = 0,0.6,0
    # colors[4,:] = 0.6, 0, 0
    # colors[5,:] = 0,0,0.6
    
    # colors[6,:] = 0,0.3,0
    # colors[7,:] = 0.3, 0, 0
    # colors[8,:] = 0,0,0.3


    # colors[6:,:] = np.repeat(np.array([[0, 0, 1]],),3, axis = 0)

    # draw_pointcloud(points, colors)
    
    return output

def get_initial_transformation(number_pixels, index, forest, data_tuple, shape, image, inv_camera_matrix, threshold = 0.05):
    forest_modes, pixels = get_random_pixel_modes(number_pixels, index, forest, data_tuple, shape, image)       
        
    test = np.zeros_like(pixels, dtype=np.float64)
    test[:,:2] = pixels[:,1:] #* pixels[:,2][:,np.newaxis]
    test[:,2] = np.ones(test.shape[0])

    
    ones = np.ones((test.shape[0], 1))
    
    for i in range(test.shape[0]):
        camera_coords = (inv_camera_matrix @ test[i,:].T)
        test[i,:] = camera_coords * pixels[i, 2] / 1000
    
    #transformation  mean <0.05 ? else resample
    transformation_matrix, dist = get_transformation_matrix(test[:,:3], forest_modes)
    if dist > 0.01:
        print("Distance too big resample")
        return get_initial_transformation(number_pixels, index, forest, data_tuple, shape, image, inv_camera_matrix)
    else:
        print('######################################## Its a match ########################################')

        return transformation_matrix


def draw_pointcloud(np_pointcloud, np_colors):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np_pointcloud)
    pcd.colors = open3d.utility.Vector3dVector(np_colors)
    open3d.visualization.draw_geometries([pcd],
                                    zoom=0.5,
                                    front=[0.5, -0.2, -1],
                                    lookat=[0, 0, 0],
                                    up=[0, -1, 0.2])