# Camera Pose Estimation
The present repository provides an approach of interferring the pose of a camera in the world coordinate system. Given a data set consisting of RGB-D images as well as camera pose matrices, the camera for a given frame can be localized. The whole process is divided into two subsequent steps:
* Finding correspondencies between 2D image pixels and 3D world coordinates
* Optimizing the camera pose for a given image

These two steps refer to the application of the *Regression Forst* and the *RANSAC optimization*, respectively. Beforehand, features are obtained using only RGB and depth values of the underlying dataset. It does not require any high level features such as SIFT or any machine learning based approach. The forest is then trained in order to learn 2D-3D correspondencies. Based on the trained forest, hypothesis camera poses are evaluated and refined using RANSAC optimization. The results are compared using the [7-scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).

The implementation is based on this beautiful [paper by Shotton et al.](https://www.microsoft.com/en-us/research/publication/scene-coordinate-regression-forests-for-camera-relocalization-in-rgb-d-images/) [1] and fully implemented in python. To get a good overview of our apporach, we suggest to have a look into  `demo.ipynb` where the following steps demonstrated and subsequently evaluated.

# Run the Project
In order to run this project you have to make sure python3.8 or python3.9 is installed on your device. To get started we provide a short shell script `setup.sh` that creates a python virtual environment with all required packages. If this worked out, we can continue with the data retrieval.

# Load Data Sets
The 7-scenes data set consists of (who guessed it?) seven different scenes with each between 2,000 and 12,000 frames. To each frame the corresponding depth map and the camera-to-world, 4×4 matrix in homogeneous coordinates is provided. The full data set was obtained using a Kinect RGB-D camera at 640×480 resolution. 

To load and clean any of the scenes, just have a look into `load_and_clean-7_scenes_dataset.ipynb` listed in the `./data` folder. Choose the desired scene and the data set will be loaded and cleaned up. If this fails, you can download the data set from the linked webpage and unzip the files manually.

# Regression Forest
Let's come to the main part of our repository. The Regression Forest is provided in the file `./source/regression_forest.py` and contains the implementation of a forest being capable of associating 2D pixel coordinates with 3D world coordinates. It requires the `./source/data_loader.py` in order to retrieve sampled image data for each tree to be trained. We further implemented the full training and test process in parallel. The respective code can be found in `./source/processing_pool.py`.  

## Feature Extraction
Each pixel that is fed into the forest results in an feature. These features are fully based on pixel and depth values. Two different types of features are currently available:
* Depth-Adaptive RGB 
* Depth

The former uses RGB and depth values of the surrounding area of the pixel. Features of the second type are using depth values only. If pixels exhibit an invalid depth value or are shifted outside the image boundary, they are not used for training or testing. We recommend to have a look at section 2.2 in [1] for more detailed information. 

## Train the Forest
To train the forest, the script `./source/train_forest.py` is utilized. By setting the required hyperparameters as well as one of the scenes, the training can be initiated. 

| Hyperparameter | Default        | 
| ------------- |:-------------:| 
|TEST_SIZE | 0.5 |
|NUM_TREES | 5 |
|TREE_MAX_DEPTH | 16 |
|NUM_TRAIN_IMAGES_PER_TREE | 500 |
|NUM_SAMPLES_PER_IMAGE | 5000   |
|NUM_PARAMETER_SAMPLES | 1024 |
|FEATURE_TYPE | DA_RGB |

The default parameters are set according to the specifications in [1]. Feel free to adjust them depending on your specific goals. Finally, the trained forest and corresponding parameters are saved.

## Test the Forest
The evaluation of the forest, is provided in the `demo.ipynb` notebook. Therefore, a number of random images is sampled from the test data set. Then corresponding image and true world coordinates are obtained, to compare the predictions later on. The forest is then tested with a batch of unseen images, or in particular their random 2D image coordinates. Eventually, the tree provides the associated 3D world coordinates predicted by each tree.

# RANSAC Optimization

# Results

# Future Thoughts
We were not able to provide this repository in form of a python package yet. This may be done in the future.
Implement Feature Type DA_RGB_DEPTH