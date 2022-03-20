import numpy as np
import open3d

from data_loader import DataLoader
from utils import millis, load_object

def draw_pointcloud(np_pointcloud):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np_pointcloud)
    open3d.visualization.draw_geometries([pcd],
                                    zoom=0.5,
                                    front=[0.5, -0.2, -1],
                                    lookat=[0, 0, 0],
                                    up=[0, -1, 0.2])

SCENE = 'stairs'
DATA_PATH = '../data'
NUM_TEST_IMAGES = 1000
NUM_SAMPLES_PER_IMAGE = 100

PREFIX = '../output/19-03-2022_20-01_stairs/'

params = load_object(f'{PREFIX}params_{SCENE}.pkl')
print(f'Loading forest trained on "{SCENE}"')

[print(f'\t{key}: {params[key]}') for key in [
    'TIMESTAMP',
    'TREE_MAX_DEPTH',
    'NUM_TRAIN_IMAGES_PER_TREE',
    'NUM_SAMPLES_PER_IMAGE',
    'NUM_PARAMETER_SAMPLES']]

forest = load_object(f'{PREFIX}trained_forest_{SCENE}.pkl')
loader = DataLoader(DATA_PATH)

# Sample points from tests data
test_set_indices = params['TEST_INDICES']
test_indices = np.random.choice(test_set_indices, NUM_TEST_IMAGES, replace = False)
images_data = loader.load_dataset(SCENE, test_indices)
p_s, w_s = loader.sample_from_data_set(
    images_data = images_data,
    num_samples = NUM_SAMPLES_PER_IMAGE)

# Evalulate tree
num_samples = NUM_TEST_IMAGES * NUM_SAMPLES_PER_IMAGE
print(f'Evaluating tree for {num_samples} samples')
start = millis()

tree_predictions = forest.evaluate(p_s, images_data)
print(f'Finished after {(millis() - start):5.1F}ms')

# Extract predictions
valid_predictions_tot = 0
predictions = np.ndarray((num_samples * params['NUM_TREES'], 3), dtype=np.float64)

for i, pred_s in enumerate(tree_predictions):
    valid_mask = ~np.any(pred_s == -np.inf, axis=1)
    w_s_valid = w_s[valid_mask]
    pred_s_valid = pred_s[valid_mask]

    valid_predictions = np.sum(valid_mask)
    errors = np.sum(np.abs(w_s_valid - pred_s_valid.copy()), axis=1)
    print(f'Tree {i}:\n\t{(100 * valid_predictions / num_samples):2.1F}% valid predictions')
    print(f'\tAverage deviation: ({np.mean(errors):1.3E} +- {np.var(errors):1.3E})m')

    predictions[valid_predictions_tot:valid_predictions_tot+valid_predictions] = pred_s_valid
    valid_predictions_tot += valid_predictions

predictions = predictions[:valid_predictions_tot]
print(f'Generating point cloud')
draw_pointcloud(predictions)