import pickle
import numpy as np

from data_loader import DataLoader

with open('trained_forest.pkl', 'rb') as inp:
    forest = pickle.load(inp)

loader = DataLoader('../data')
images_data = loader.load_dataset('heads', (0, 100))
p_s, w_s = loader.sample_from_data_set(
    images_data = images_data,
    num_samples = 5000)

pred_s = forest.evaluate(p_s, images_data)

valid_mask = ~np.any(pred_s == -np.inf, axis=1)
w_s_valid = w_s[valid_mask]
pred_s_valid = pred_s[valid_mask]

print(f'pred_s: {pred_s}')
print(f'pred_s_valid: {pred_s_valid}')
print(f'w_s_valid: {w_s_valid}')
print(f'w_s_valid - pred_s_valid: {w_s_valid - pred_s_valid}')

errors = np.sum(np.abs(w_s_valid - pred_s_valid), axis=1)
print(f'Deviations: var: {np.var(errors):3.3E}  mean: {np.mean(errors):3.3E}')


