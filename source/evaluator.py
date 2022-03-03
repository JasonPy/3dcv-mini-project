import numpy as np

# TODO: make sure to provide the 3D points of the following form: np.array([[1,2,3],[1,2,3], ...])


class Evaluator:
    """
    This class represents a tool to evaluate the percentage of correctly predicted poses.
    Thereby the predictions are compared to the ground truth according to the translational and angular error.
    """

    def __init__(self, translational_error_threshold: float = 5., angular_error_threshold: float = 5.):
        """
        Parameters
        ----------
        translational_error_threshold: float
            distance in cm 
        angular_error_threshold: float
            angle in degrees
        """
        self.translational_error_threshold = translational_error_threshold
        self.angular_error_threshold = angular_error_threshold


    def evaluate(self, poses: np.array, ground_truth: np.array) -> float:
        """
        Calculates  the percentages of 'correctly' classified test frames.
        A pose must be within 5cm translational error and 5° angular error of the ground truth to be
        classified as correct.

        Parameters
        ----------
        poses: np.array
            the poses predicted by some algorithm as 3D world coordinates
        ground_truth: np.array
            the true poses as 3D world coordinates

        Returns
        -------
        metric: float
            the percentage of poses classified as 'correct'
        """

        error_angular = self.get_angular_error(poses, ground_truth)
        error_translational = self.get_translational_error(poses, ground_truth)

        print(error_angular)

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
        Calculate the angle between two sets of 3D points.

        Parameters
        ----------
        poses: np.array
            the poses predicted by some algorithm as 3D world coordinates
        ground_truth: np.array
            the true poses as 3D world coordinates

        Returns
        -------
        angular_error: np.array
            angular error between corresponding 3D world points
        """

        # calculate the dot product and normalize by their length 
        dot_product = np.sum(poses * ground_truth, axis=1) / (np.linalg.norm(poses, axis=1) * np.linalg.norm(ground_truth, axis=1))
        
        # clip result of dot-product to prevent runtime issues when encountering colinear vectors
        dot_product = np.clip(dot_product, -1, 1)

        # calculate angle between vectors and transform to degrees
        return np.rad2deg(np.arccos(dot_product))


    def get_translational_error(self, poses: np.array, ground_truth: np.array) -> np.array:
        """
        The translational error is defined as the distance between two points.

        Parameters
        ----------
        poses: np.array
            the poses predicted by some algorithm as 3D world coordinates
        ground_truth: np.array
            the true poses as 3D world coordinates

        Returns
        -------
        translational_error: np.array
            translational error between corresponding 3D world points
        """
        return np.sqrt(np.sum((poses - ground_truth)**2, axis=1))
