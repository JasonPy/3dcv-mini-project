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
from processing_pool_ransac import ProcessingPool 

INVALID_DEPTH_VALUE = 65535


def initialize_hypotheses(image, forest, number_pixels, k, data_tuple, index, inv_camera_matrix, tqdm_pbar):
    """
    Generate k hypotheses based on n (number_pixels) pixels, using the kabsch algorithm

    Args:
        image (ndarray): is used to retrieve the depth information
        number_pixels (int): the number of pixel that are used for the generation
        k (int): The number of hypotheses
        index : the position of the image in the data tuple
        
    Returns: 
        array (4,4,k) : initial hypotheses stored along axis = 2
    """
    shape = image.shape
        
    #generated hypotheses
    hypotheses = np.zeros((4,4,k))
    
    x = np.linspace(0, shape[0] - 1, shape[0])
    y = np.linspace(0, shape[1] - 1, shape[1])
    
    #switch x,y for method call since image is transposed
    xx, yy = np.meshgrid(x, y, indexing="ij")
    
    depths = image[:,:,3]
    valid_depths_mask = ~((depths == INVALID_DEPTH_VALUE) | (depths == 0))
    
    valid_x = xx[valid_depths_mask]
    valid_y = yy[valid_depths_mask]
    
    for step in range(k):
        forest_modes, random_pixels = get_random_pixel_modes(number_pixels, index, forest, data_tuple, shape, image, valid_x, valid_y)
        pix = random_pixels[:,:2].astype(np.int32)
        depths = data_tuple[1][index][pix[:,0], pix[:,1]]
        
        x_y_h = np.zeros_like(random_pixels, dtype=np.float64)
        x_y_h[:,:2] = random_pixels[:,:2] #* pixels[:,2][:,np.newaxis]
        x_y_h[:,2] = np.ones(x_y_h.shape[0])
        pixe = np.hstack((pix, np.ones(pix.shape[0], dtype=np.int32)[:, np.newaxis]))
        # calculate for each pixel the camera scene coordinates
        for i in range(x_y_h.shape[0]):
            camera_coords = (inv_camera_matrix @ x_y_h[i,:].T)
            x_y_h[i,:] = camera_coords * random_pixels[i, 2] / 1000

        hypothesis = get_transformation_matrix(x_y_h[:,:3], forest_modes)
         
        hypotheses[:,:,step] = hypothesis
        tqdm_pbar.update(1)

    return hypotheses



def get_random_pixel_modes(n, index, forest, data_tuple, shape, image, valid_xs, valid_ys):
    """
    Generate n random pixels and evaluate each pixel with the given forest.    
    
    Args:
        n (int): the number of random pixels
        index (int) : the index of the image in the data_tuple
        forest (forest): Forest for the evaluation
        data_tuple (Tuple(array, array, array)): First index contains rgb images, second index depth images, third positions contains poses
        shape (_type_): _description_
        image (_type_): _description_
        valid_xs : valid x pixel values to sample from
        valid_ys : valid y pixel values to sample from

    Returns:
        modes : n random modes
        pixles : n random pixles (x, y)
    """
    
   
    random_position = np.random.choice(valid_xs.shape[0], size= n, replace=True)
    pixels_xs = valid_xs[random_position]
    pixel_ys = valid_ys[random_position]
    
    random_pixels = np.hstack((np.repeat(index, n)[:,np.newaxis], pixels_xs[:, np.newaxis], pixel_ys[:,np.newaxis])).astype(np.int32)
    random_pixels_depths = image[random_pixels[:,1], random_pixels[:,2], 3]
    # depths = data_tuple[1][random_pixels[:,1], random_pixels[:,2]]
    
    predictions = forest.evaluate(random_pixels, data_tuple)

    
    modes, pixel_mask = sample_n_random_modes(predictions)
    
    if modes is None:
        return get_random_pixel_modes(n, index, forest, data_tuple, shape, image, valid_xs, valid_ys)

    random_pixels = random_pixels[pixel_mask,:]
    modes = modes[:3, :]
    random_pixels = random_pixels[:3,:]

    
    x_y_d = np.zeros(random_pixels.shape)
    x_y_d[:,:2] = random_pixels[:,1:]
    x_y_d[:,2] = random_pixels_depths[pixel_mask][:3]
       
    return modes, x_y_d


def energy_function(predicted_positions, pixel, camera_hypothesis):
    # energy_function(pixel_modes, camera_coords, hypotheses[:,:,valid_index])
    """
    Calculate the energy for a given hypothesis based on a top hat error functino with a width defined by top_hat_error_width

    Args:
        predicted_position (_type_): predicted position
        pixel (_type_): 3d position, camera space
        camera_hypothesis (_type_): _description_
        inv_camera_matrix: the inverse camera matrix
    """

    t = np.append(pixel,1)
    
    top_hat_error_width = 2
    min_distance = np.min(np.linalg.norm(predicted_positions[:,:3] - ((camera_hypothesis @ t).T)[:3], axis = 1))
        
    return min_distance
 

def optimize(forest, image, k, index, data_tuple, inv_camera_matrix, number_pixels = 10, batch_size = 500):
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
    init_time = time.time()
    hypotheses = initialize_hypotheses(image, forest, number_pixels, k, data_tuple, index, inv_camera_matrix, pbar)
    print(time.time() - init_time)
    
    pbar.clear()
    faulthandler.enable()
    pixel_inliers = {}
    #store for each pixel the modes
    pixel_inliers_modes = {}
    errors = {}
    
    valid_hypotheses = len(hypotheses)
    valid_hypotheses_indices = np.arange(0, hypotheses.shape[2])

    #initialize for each hypothese a list
    for element in valid_hypotheses_indices:
          pixel_inliers[element] = []
          pixel_inliers_modes[element] = []

    energies = np.zeros(len(valid_hypotheses_indices))
    total_points = 0
    # processing_pool = ProcessingPool(inv_camera_matrix, energy_function_worker)
    
    # work_queue = processing_pool.queue_work
    # result_queue = processing_pool.queue_result
    forest_time = 0
    while len(valid_hypotheses_indices) > 1:
        #sample random set B of image coordinates
        t0 = time.time()
        pixel_batch = sample_pixel_batch(batch_size, image.shape, index, image[:,:,3])

        valid_depth_mask = ~((image[pixel_batch[:,1], pixel_batch[:,2],3] == INVALID_DEPTH_VALUE) | (image[pixel_batch[:,1], pixel_batch[:,2],3] == 0))
        pixel_batch = pixel_batch[valid_depth_mask,:]
        print(f"sample pixel {time.time() - t0}")
        t1 = time.time()
        modes = forest.evaluate(pixel_batch, data_tuple)
        print(f"Forest time {time.time() - t1}")
        forest_time += time.time() - t1
        modes = np.asarray(modes)
        
        invalid = 0
        t2 = time.time()

        for pixel_index in range(pixel_batch.shape[0]):                    
            depth = image[pixel_batch[pixel_index, 1], pixel_batch[pixel_index, 2], 3]
            pixel = np.array([pixel_batch[pixel_index, 1], pixel_batch[pixel_index, 2], depth]).T
            
            pixel_modes = modes[:,pixel_index,:]
            tmp = modes[:,pixel_index, :]
            #check if more than 0 modes are not none and remove them
            inv_c = ~np.isfinite(pixel_modes)
            if np.sum(~np.isfinite(pixel_modes)[:,0], axis = 0) < pixel_modes.shape[0]:
                pixel_modes = pixel_modes[np.isfinite(pixel_modes)[:,0],:]
            #all modes are none skip pixel
            elif np.any(~np.isfinite(pixel_modes)):
                invalid += 1
                continue

            
            pixel_modes = np.append(pixel_modes, np.ones((pixel_modes.shape[0], 1)), axis = 1)
            p = pixel
            p[2] = 1
            camera_coords = inv_camera_matrix @ p * depth / 1000
            
            camera_coords_h = np.append(camera_coords, 1)[np.newaxis, :, np.newaxis]
            camera_coords_h = np.repeat(camera_coords_h, hypotheses[:,:,valid_hypotheses_indices].shape[2], axis = 2)
            
            #matrix multiplication each column corresponds to one matrix multiplication
            world = np.sum(hypotheses[:,:,valid_hypotheses_indices] * camera_coords_h, axis = 1)[:3,:].T
            world = world[:,:,np.newaxis]
            pixel_modes_tmp = np.swapaxes(pixel_modes[:,:3][np.newaxis, :,:],1, 2)
            a = np.min(np.sqrt(np.sum((world - pixel_modes_tmp)**2, axis = 1)), axis = 1) > 0.1
            energies[valid_hypotheses_indices] += a

            for h_index in valid_hypotheses_indices[~a]:
                pixel_inliers[h_index].append(np.array([index, pixel[0], pixel[1]]))
                pixel_inliers_modes[h_index].append(tmp)

        print(f'Loop time {time.time() - t2}')
        total_points += (batch_size - invalid)
                
        sorted_energy_indices = np.argsort(energies[valid_hypotheses_indices])       
        
        half_len = int(len(sorted_energy_indices) // 2)
        lower_half_indices = valid_hypotheses_indices[sorted_energy_indices[:half_len]]
        
        #check if hypothesis has inlier
        if len(valid_hypotheses_indices) == len(energies):
            mask = energies[lower_half_indices] < pixel_batch.shape[0] - invalid
            lower_half_indices = lower_half_indices[mask]
        
        
       
        #refine hypotheses based on inlieres and kabsch
        i = 0
        
        # pbar = tqdm(total = lower_half_indices.shape[0], desc = "optimize hypotheses")
        t1 = time.time()
        for hypothesis_index in lower_half_indices:
            pixels = np.asarray(pixel_inliers[hypothesis_index])
            modes = pixel_inliers_modes[hypothesis_index]
            modes = np.swapaxes(np.asarray(modes), 0, 1)
            # modes = forest.evaluate(pixels, data_tuple)
            if pixels.shape[0] == 0:
                print("No NO NO inlier")
                continue
            
            # modes = forest.evaluate(pixels, data_tuple)
            # check24 = np.sum(~(test == modes))
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
            camera__scene_coord = camera__scene_coord[mask,:]
            
            hypotheses[:,:,hypothesis_index] = get_transformation_matrix(camera__scene_coord, random_modes)           

        #     pbar.update(1)
        # pbar.close()
        print(f'\n optimize time {time.time() - t1}')

        valid_hypotheses_indices = lower_half_indices
    print(f'################# forest time : {forest_time}')
    return hypotheses[:,:,valid_hypotheses_indices[0]], energies[valid_hypotheses_indices[0]]
        
    

def sample_pixel_batch(batch_size, image_shape, index, depth):

    coordinate_range = np.arange(image_shape[0] * image_shape[1], dtype=np.int32)
    i = 0
    # p_s = np.zeros((batch_size, 3), dtype=np.int32)
    # while i < batch_size:
    x_y_s = np.random.choice(coordinate_range, batch_size, replace=False)
    x_s = x_y_s // 480
    y_s = x_y_s % 480
        
        # if depth[x_s, y_s] != 65535 and depth[x_s, y_s] != 0:
        #     p_s[i,:] = index, x_s, y_s
        #     i += 1
    return np.stack((np.full(x_s.shape, index), x_s, y_s)).T

def sample_n_random_modes(modes, n = 3):
    """ Evalutes the modes. Checks if the modes are valid at least for n pixels. 

    Args:
        modes : predicted modes

    Returns:
        output : the valid modes as array (valid_pixels, 3)
        mask : boolean mask indicates which pixels have a valid mode
    """
    modes = np.asarray(modes)
    
    modes_mask = np.isfinite(modes[:, :, 0])
    
    #indicates the columns with valid modes
    valid_columns = np.any(modes_mask, axis = 0)
    
    #less than n pixel modes are valid
    if np.sum(valid_columns, axis = 0) < n:
        return None, None
    
    upper_limit = modes.shape[0]
    
    output = np.zeros((np.sum(valid_columns, axis = 0), 3))
   
    #iterate over valid_columns and sample a random tree mode
    for i, valid_column in enumerate(np.where(valid_columns == 1)[0]):
        #g
        indices = np.where(modes_mask[:,valid_column] == 1)[0]
        random_tree_index = np.random.choice(indices, 1)
        mode = modes[random_tree_index, valid_column, :]
        output[i,:] = mode
    
    return output, np.any(modes_mask, axis = 0)

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
    
    # q = (output @ np.append(a, ones, axis = 1).T).T
    # mean_distance = np.mean(np.linalg.norm(b - q[:, :3], axis = 1))
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

def energy_function_worker(inv_camera_matrix, work_data):
    predicted_positions, pixel, camera_hypothesis, hypotheses_index, image_index = work_data
    
    tmp_pixel = np.zeros_like(pixel, dtype=np.float64)
    tmp_pixel[:2] = pixel[:2] 
    tmp_pixel[2] = 1
    # tmp_pixel[:3] = tmp_pixel[:3] / 1000.0
    t = inv_camera_matrix @ tmp_pixel
    t= t * pixel[2] / 1000
    t = np.append(t,1)
    
    top_hat_error_width = 2
    min_distance = np.min(np.linalg.norm(predicted_positions[:,:3] - ((camera_hypothesis @ t))[:3]))
    
    return (hypotheses_index, np.array([image_index, pixel[0], pixel[1]]), int(~(min_distance < 0.1)))

def draw_pointcloud(np_pointcloud, np_colors):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np_pointcloud)
    pcd.colors = open3d.utility.Vector3dVector(np_colors)
    open3d.visualization.draw_geometries([pcd],
                                    zoom=0.5,
                                    front=[0.5, -0.2, -1],
                                    lookat=[0, 0, 0],
                                    up=[0, -1, 0.2])