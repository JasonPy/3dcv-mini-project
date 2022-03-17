import ransac

import pickle
import numpy as np

from data_loader import DataLoader

with open('trained_forest_1024.pkl', 'rb') as inp:
    forest = pickle.load(inp)
    
inp.close()

loader = DataLoader('../data')
images_data = loader.load_dataset('heads', (0, 3))

rgb_image = images_data[0][0]

depth_image = images_data[1][0]
depth_image = depth_image[:, :, np.newaxis]

image = np.concatenate((rgb_image, depth_image), axis = 2)
data_tuple = (images_data[0][0], images_data[0][1], images_data[0][2])

hypa, energie = ransac.optimize(forest, image, 10, images_data)
print(hypa)