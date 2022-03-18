import ransac

import pickle
import numpy as np

from data_loader import DataLoader
import sys


with open('trained_forest_stairs.pkl', 'rb') as inp:
    forest = pickle.load(inp)
    
inp.close()

loader = DataLoader('../data')
images_data = loader.load_dataset('stairs', range(0,10))

rgb_image = images_data[0][0]

depth_image = images_data[1][0]
depth_image = depth_image[:, :, np.newaxis]

image = np.concatenate((rgb_image, depth_image), axis = 2)
data_tuple = (images_data[0][0], images_data[0][1], images_data[0][2])

hypa, energy = ransac.optimize(forest, image, 1024, images_data)
np.savetxt("energy", np.array([energy]))
np.savetxt("camera_pose", hypa)