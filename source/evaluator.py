import numpy as np

# TODO: make sure to provide the 3D poses in an numpy array with 4x4 matrices np.array([[4x4], [4x4], ...])


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
        translational_error = np.sum(np.sqrt((T_pos - T_gt)**2), axis=1)

        return translational_error
