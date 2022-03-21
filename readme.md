# Camera Pose Estimation
The present repository provides an approach of interferring the pose of a camera in the world coordinate system. Given a data set consisting of RGB-D images as well as camera pose matrices, the camera for a given frame can be localized. The whole process is divided into two subsequent steps:
* Finding correspondencies between 2D image pixels and 3D world coordinates
* Optimizing the camera pose for a given image

These two steps refer to the application of the *Regression Forst* and the *RANSAC optimization*, respectively. Beforehand, features are obtained using only RGB and depth values of the underlying dataset. It does not require any high level features such as SIFT or any machine learning based approach. The forest is then trained in order to learn 2D-3D correspondencies. Based on the trained forest, hypothesis camera poses are evaluated and refined using RANSAC optimization. The results are compared using the [7-scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).

The implementation is based on this beautiful [paper by Shotton et al.](https://www.microsoft.com/en-us/research/publication/scene-coordinate-regression-forests-for-camera-relocalization-in-rgb-d-images/) and fully implemented in python.

# Run the Project
In order to run this project you have to make sure python3.8 or python3.9 is installed on your device. To get started we provide a short shell script `setup.sh` that creates a python virtual environment with all required packages. If this worked out, we can continue with the data retrieval.

# Load Data Sets
The 7-scenes data set consists of (who guessed it?) seven different scenes with each between $2,000$ and $12,000$ frames. To each frame the corresponding depth map and the camera-to-world, 4×4 matrix in homogeneous coordinates is provided. The full data set was obtained using a Kinect RGB-D camera at 640×480 resolution. 

To load and clean any of the scenes, just have a look into `load_and_clean-7_scenes_dataset.ipynb` listed in the `./data` folder. Choose the desired scene and the data set will be loaded and cleaned up.

# Regression Forest
Let's come to the main part of our implementation. 

## Train the Forest
## Test the Forest

# RANSAC Optimization

# Evaluate Results
