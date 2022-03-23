import numpy as np

def get_valid_predictions(tree_predictions):
    valid_predictions_tot = 0
    predictions = np.ndarray((tree_predictions[0].shape[0] * len(tree_predictions), 3), dtype=np.float64)

    for pred in tree_predictions:
        valid_mask = ~np.any(pred == -np.inf, axis=1)
        valid_predictions = np.sum(valid_mask)
        predictions[valid_predictions_tot:valid_predictions_tot+valid_predictions] = pred[valid_mask]
        valid_predictions_tot += valid_predictions
    return predictions[:valid_predictions_tot]

def get_prediction_error(tree_predictions, ground_truth):
    errors = []

    for pred in tree_predictions:
        valid_mask = ~np.any(pred == -np.inf, axis=1)
        errors.append(np.linalg.norm(ground_truth[valid_mask] - pred[valid_mask], axis=1))
    return errors