from argparse import ArgumentError
from typing import Union, Tuple, List
import numpy as np

from feature_extractor import FeatureExtracor, FeatureType

class SampleHolder:
    def __init__(self, feature_extractor: FeatureExtracor, p_s: np.array, w_s: np.array):
        self.feature_extractor = feature_extractor
        self.p_s = p_s
        self.w_s = w_s

    def __getitem__(self, argument: Union[int, np.array]) -> Union[Tuple[np.array, np.array], 'SampleHolder']:
        # Yes, the type-hint for the SampleHolder return-type has to be in quotes:
        # https://www.python.org/dev/peps/pep-0484/#forward-references

        if isinstance(argument, np.array):

            if argument.shape[0] != len(self.p_s):
                raise ArgumentError(f'Mask of length {argument.shape[0]} cannot be applied to '
                                    f'SampleHolder of size {len(self.p_s)}')

            p_s_sliced = self.p_s[argument]
            w_s_sliced = self.w_s[argument]
            return SampleHolder(self.feature_extractor, p_s_sliced, w_s_sliced)

        elif isinstance(argument, int):
            return (self.p_s[argument], self.w_s[argument])

    def is_empty(self):
        return len(self.p_s) == 0

    def get_features_with_parameters(self, params: any, feature_type: FeatureType) -> np.array:
        feature_extractor_vec = np.vectorize(self.feature_extractor.get_feature, exclude=['params'])
        return feature_extractor_vec(self.p_s, params, feature_type)

class DataHolder:
    def __init__(self, samples: List[SampleHolder] = []):
        self.samples = samples

    def add_samples(self, sample: SampleHolder):
        self.samples.append(sample)

    def __getitem__(self, masks: List[np.array]) -> 'DataHolder':
        if (len(masks) != len(self.samples)):
            raise ArgumentError('Number of masks does not match number of SampleHolders in DataHolder')

        data_masked = DataHolder()
        for i in range(len(self.samples)):
            samples_masked = self.samples[i][masks[i]]
            if not samples_masked.is_empty():
                data_masked.add_samples(samples_masked)

        return data_masked


def sample_from_feature_extractor(feature_extractor: FeatureExtracor, num_samples: int) -> SampleHolder:
    (p_s, w_s) = feature_extractor.generate_data_samples(num_samples)
    return SampleHolder(feature_extractor, p_s, w_s)

def sample_from_data_set(images_rgb: np.array, images_depth: np.array, camera_poses: np.array, num_samples: int) -> DataHolder:
    """
    Samples the specified number of samples from each of the provided images.
    
    Parameters
    ----------
    images_rgb: np.array
        The RGB values for the images as returned by create_numpy_dataset
    images_depth: np.array
        The depth values for the images as returned by create_numpy_dataset
    camera_poses: np.array
        The parsed camera poses as returned by create_numpy_dataset
    num_samples: int
        The number of samples to draw from each image
    """
    data_holder = DataHolder()

    for i in range(images_rgb.shape[0]):
        feature_extractor = FeatureExtracor(
            depth_data=images_depth[i],
            rgb_data=images_rgb[i],
            camera_pose=camera_poses[i])
        samples = sample_from_feature_extractor(feature_extractor, num_samples)
        data_holder.add_samples(samples)
    
    return data_holder

