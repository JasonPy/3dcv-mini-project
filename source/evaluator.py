from typing import List
import numpy as np

class PoseEvaluator:
    """
    This class represents a tool to evaluate the percentage of correctly predicted poses.
    Thereby the predictions are compared to the ground truth according to the translational and angular error.
    """

    def __init__(self, translational_error_threshold: float = 5., angular_error_threshold: float = 5.):
        """
        Consider that the pose matrices are in meters. The 'translational_error_threshold' is 
        therefore casted from cm to m. 

        Parameters
        ----------
        translational_error_threshold: float
            distance in cm
        angular_error_threshold: float
            angle in degrees
        """
        self.translational_error_threshold = translational_error_threshold / 100. # to m
        self.angular_error_threshold = angular_error_threshold


    def evaluate(self, poses: np.array, ground_truth: np.array) -> float:
        """
        Calculates  the percentages of 'correctly' classified test frames.
        A pose must be within 5cm translational error and 5° angular error of the ground truth to be
        classified as correct.

        Parameters
        ----------
        poses: np.array
            the poses predicted by some algorithm as 4x4 matrices
        ground_truth: np.array
            the true poses as 4x4 matrices

        Returns
        -------
        metric: float
            the percentage of poses classified as 'correct'
        """

        error_angular = self.get_angular_error(poses, ground_truth)
        error_translational = self.get_translational_error(poses, ground_truth)

        # check which of the values are below the corresponding threshold
        inliers_angular = error_angular <= self.angular_error_threshold
        inliers_translational = error_translational <= self.translational_error_threshold

        # evaluate for which poses translational and angular error are below threshold
        total_inliers = np.logical_and(inliers_angular, inliers_translational)

        # return percentage of correct poses
        metric = np.sum(total_inliers) / poses.shape[0]
        return metric


    def get_angular_error(self, poses: np.array, ground_truth: np.array) -> np.array:
        """
        Calculate the angle between two sets of poses.

        Parameters
        ----------
        poses: np.array
            the poses predicted by some algorithm as an array of 4x4 matrices
        ground_truth: np.array
            the true poses as an array of 4x4 matrices

        Returns
        -------
        angular_error: np.array
            angular error between corresponding poses and ground truth
        """

        # obtain the rotation matrices from the pose matrices
        R_pos = poses[:,:3,:3]
        R_gt = ground_truth[:,:3,:3]

        # compute the difference matrix, representing the difference rotation
        R_diff = np.matmul(R_pos, np.transpose(R_gt, axes=(0,2,1)))

        # get the angle theta of difference rotations
        theta = (np.trace(R_diff, axis1=1, axis2=2) - 1) / 2
        angular_error = np.rad2deg(np.arccos(np.clip(theta, -1, 1)))

        return angular_error


    def get_translational_error(self, poses: np.array, ground_truth: np.array) -> np.array:
        """
        The translational error is defined as the distance between two points.

        Parameters
        ----------
        poses: np.array
            the poses predicted by some algorithm as array of 4x4 matrices
        ground_truth: np.array
            the true poses as array of 4x4 matrices

        Returns
        -------
        translational_error: np.array
            translational error between corresponding poses
        """

        # obtain translation vectors from poses
        T_pos = poses[:,:3,3]
        T_gt = ground_truth[:,:3,3]

        # calculate translational error
        translational_error = np.linalg.norm(T_pos- T_gt, axis=1)

        return translational_error


class SceneCoordinateEvaluator:
    """
    This class handles the evaluation of predicted 3D scene coordinates.
    """

    def __init__(self):
        pass

    def get_valid_predictions(self, tree_predictions) -> np.array:
        """
        Evaluates 3D world coordinates in terms of invalid values (-np.inf)
        and removes them from the initial predictions.

        Parameters
        -------
        tree_predictions: List[np.array]
            World coordinate predictions for each tree

        Returns
        -------
        predictions: np.array
            The predictions that are valid
        """
        valid_predictions_tot = 0
        predictions = np.ndarray((tree_predictions[0].shape[0] * len(tree_predictions), 3), dtype=np.float64)

        for pred in tree_predictions:
            valid_mask = ~np.any(pred == np.inf, axis=1)
            valid_predictions = np.sum(valid_mask)
            predictions[valid_predictions_tot:valid_predictions_tot+valid_predictions] = pred[valid_mask]
            valid_predictions_tot += valid_predictions
        return predictions[:valid_predictions_tot]

    def get_prediction_error(self, tree_predictions, ground_truth) -> List:
        """
        Calculates the error for each predicted world coordinate
        with respect to the ground truth. The L2-norm is utilized.

        Parameters
        -------
        tree_predictions: List[np.array]
            World coordinate predictions for each tree
        ground_truth: np.array
            True 3D world coordinates

        Returns
        -------
        errors: List[np.array]
            Error for all coordinates in each tree
        """
        errors = []

        for pred in tree_predictions:
            valid_mask = ~np.any(pred == np.inf, axis=1)
            errors.append(np.linalg.norm(ground_truth[valid_mask] - pred[valid_mask], axis=1))
        return errors
