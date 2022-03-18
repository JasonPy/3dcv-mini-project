# import pickle
import numpy as np
import open3d

from data_loader import DataLoader

# def save_object(obj, filename):
#     with open(filename, 'wb') as outp:  # Overwrites any existing file.
#         pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

DATA_PATH = '../data'
SCENE = 'chess'

NUM_IMAGES = 100
NUM_SAMPLES_PER_IMAGE = 5000

loader = DataLoader(DATA_PATH)
image_indices = np.random.choice(np.arange(loader.get_dataset_length(SCENE)), NUM_IMAGES)
# image_indices = [20, 80]
images_data = loader.load_dataset(SCENE, image_indices)
sample_points = loader.sample_from_data_set(images_data, NUM_SAMPLES_PER_IMAGE)

np_point_cloud = sample_points[1]
o3d_point_cloud = open3d.utility.Vector3dVector(np_point_cloud)

# print(np_point_cloud)

pcd = open3d.geometry.PointCloud()
pcd.points = o3d_point_cloud
open3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[0, 0, 0],
                                  up=[-0.0694, -0.9768, 0.2024])




# forest = RegressionForest(
#     num_trees = NUM_TREES,
#     max_depth = TREE_MAX_DEPTH,
#     feature_type = FeatureType.DA_RGB,
#     num_param_samples = NUM_PARAMETER_SAMPLES,
#     param_sampler = param_sampler,
#     objective_function = objective_reduction_in_variance)

# forest.train(
#     data_dir = DATA_PATH,
#     scene_name = SCENE,
#     train_image_indices = train_indices,
#     num_images_per_tree = NUM_TRAIN_IMAGES_PER_TREE,
#     num_samples_per_image = NUM_SAMPLES_PER_IMAGE)

# # save used parameters
# params = {
#   "TIMESTAMP": TIMESTAMP,
#   "SCENE": SCENE,
#   "TEST_SIZE": TEST_SIZE,
#   "NUM_TREES": NUM_TREES,
#   "TREE_MAX_DEPTH": TREE_MAX_DEPTH,
#   "NUM_TRAIN_IMAGES_PER_TREE": NUM_TRAIN_IMAGES_PER_TREE,
#   "NUM_SAMPLES_PER_IMAGE": NUM_SAMPLES_PER_IMAGE,
#   "NUM_PARAMETER_SAMPLES": NUM_PARAMETER_SAMPLES,
#   "SPLIT_MASK": [train_indices, test_indices],
# }

# if forest.is_trained:
#     save_object(forest, f'trained_forest_{SCENE}.pkl')
#     save_object(params, f'params_{SCENE}.pkl')
#     print(f'Done training forest of scene {SCENE}.')
# else:
#     print(f'Training interrupted!')