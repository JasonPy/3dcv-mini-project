from argparse import ArgumentError
from random import sample
from typing import Union, Tuple, List
import numpy as np

from feature_extractor import FeatureExtracor, FeatureType

class Sample:
    def __init__(self, feature_extractor: FeatureExtracor, p: np.array, w: np.array):
        """
        Simple wrapper holding a simple data point and associated FeatureExtractor

        Parameters
        ----------
        feature_extractor: FeatureExtractor
            The feature_extractor reference holding the image data associated with this
            data point
        p: np.array
            This sample's input feature (image coordinates)
        w: np.array
            This sample's target response (world coordinates)
        """
        self.feature_extractor = feature_extractor
        self.p = p
        self.w = w

    def get_feature_with_parameters(self, params: any, feature_type: FeatureType) -> bool:
        """
        Evaluate the feature extractor at this data point with the given parameters
        and FeatureType
        
        Parameters
        ----------
        params: any
            The parameters passed to the FeatureExtractor
        feature_type: FeatureType
            The type of feature to evaluate
        """
        return self.feature_extractor.get_feature(self.p, params, feature_type)

class SampleHolder:
    def __init__(self, feature_extractor: FeatureExtracor, p_s: np.array, w_s: np.array):
        """
        A holder, which holds a list of data points sharing the same FeatureExtractor (image)
        
        We don't save a List[Sample] to more efficiently work on a collection of samples
        which share the same FeatureExtractor

        Parameters
        ----------
        feature_extractor: FeatureExtractor
            The feature_extractor reference holding the image data associated with the
            data points
        p_s: np.array
            Array of input features of the data points
        w_s: np.array
            Array of target responses of the data points
        """
        self.feature_extractor = feature_extractor
        self.p_s = p_s
        self.w_s = w_s

    def __len__(self):
        return len(self.p_s)

    # Yes, the type-hint for the SampleHolder return-type has to be in quotes:
    # https://www.python.org/dev/peps/pep-0484/#forward-references
    def __getitem__(self, argument: Union[int, np.array]) -> Union[Sample, 'SampleHolder']:
        """
        Python magic function which implements the array access operator:
            "object.__getitem__(a) == object[a]"

        Supports accessing a single elemement (type(argument) == int) returning a Sample
        as well as masking the complete SampleHolder with a mask array (type(argument) == np.array)

        Parameters
        ----------
        argument: int
            The index of the sample to return
        argument: np.array
            np.array describing a boolean mask for this sample-list

        Returns
        -------
        arg == int:
            A single Sample
        arg == np.array
            A SampleHolder with containing only the samples on whose index the mask evaluates to true
        """
        
        if isinstance(argument, np.array):

            if argument.shape[0] != len(self.p_s):
                raise ArgumentError(f'Mask of length {argument.shape[0]} cannot be applied to '
                                    f'SampleHolder of size {len(self.p_s)}')

            p_s_sliced = self.p_s[argument]
            w_s_sliced = self.w_s[argument]
            return SampleHolder(self.feature_extractor, p_s_sliced, w_s_sliced)

        elif isinstance(argument, int):
            return Sample(self.feature_extractor, self.p_s[argument], self.w_s[argument])

    def is_empty(self):
        return len(self.p_s) == 0

    def get_features_with_parameters(self, params: any, feature_type: FeatureType) -> np.array:
        """
        Vectorized version of the Sample.get_feature_with_parameters
        """
        feature_extractor_vec = np.vectorize(self.feature_extractor.get_feature, exclude=['params'])
        return feature_extractor_vec(self.p_s, params, feature_type)

class DataHolder:
    def __init__(self, samples: List[SampleHolder] = []):
        """
        Holds a set of SampleHolders and allows vectorized masking

        Parameters
        ----------
        samples: List[SampleHolder]
            The initial list of SampleHolders to construct this DataHolder with
        """
        self.samples = samples

    def __len__(self):
        """
        Returns
        -------
        The number of Samples in this DataHolder
        """
        return sum([len(sample_holder) for sample_holder in self.samples])

    def __getitem__(self, argument: Union[int, List[np.array]]) -> Union['DataHolder', SampleHolder]:
        """
        Array access operator. Supports element access (indexed by SampleHolder)
        and masking of complete list of Samples (indexed by nested masks)

        Parameters
        ----------
        argument == int
            The SampleHolder index

            Returns
            -------
            SampleHolder

        argument == List[np.array]
            The nested masks to apply to this DataHolder. len(argument) should be == len(self.samples)
            len(argument[i]) should be == len(self[i])

            Returns
            -------
            DataHolder with recursivley applied masks. Empty SampleHolders are pruned

        """
        if isinstance(argument, int):
            return self.samples[argument]
        
        elif isinstance(argument, np.array):
            masks = argument
            if (len(masks) != len(self.samples)):
                raise ArgumentError('Number of masks does not match number of SampleHolders in DataHolder')

            data_masked = DataHolder()
            for i in range(len(self.samples)):
                samples_masked = self.samples[i][masks[i]]
                if not samples_masked.is_empty():
                    data_masked.add_samples(samples_masked)

            return data_masked
    
    def add_samples(self, sample: SampleHolder) -> None:
        """
        Add a SampleHodler to this DataHolder
        TODO: Maybe merge with existing SampleHolder on equal FeatureExtractor?

        Parameters
        ----------
        sample: SampleHolder
            The list of samples to add
        """
        self.samples.append(sample)

    def get_all_sample_data_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return all sample data points (input feature and target response)
        in all SampleHolders.
        
        Returns
        -------
        (p_s, w_s) : Tuple[np.ndarray, np.ndarray]
            p_s: np.ndarray
                The input_feature data points (image coordinates)
            w_s: np.ndarray
                The target_response data points (3d world coordinates)
        """
        total_samples = len(self)
        single_p = self.samples[0].p_s[0]
        single_w = self.samples[0].w_s[0]

        shape_p_total = (total_samples, *single_p.shape)
        shape_w_total = (total_samples, *single_w.shape)
        p_s_total = np.ndarray(dtype=single_p.dtype, shape=shape_p_total)
        w_s_total = np.ndarray(dtype=single_w.dtype, shape=shape_w_total)

        idx_total_start = 0
        for i in range(len(self.samples)):
            sample_holder = self.samples[i]
            idx_total_end = idx_total_start + len(sample_holder)
            p_s_total[idx_total_start:idx_total_end] = sample_holder.p_s
            w_s_total[idx_total_start:idx_total_end] = sample_holder.w_s
            idx_total_start = idx_total_end

        return (p_s_total, w_s_total)

    def get_features_with_parameters(self, params: any, feature_type: FeatureType) -> np.array:
        """
        Nested version of the SampleHolder.get_feature_with_parameters
        """
        masks = []
        for sample_holder in self.samples:
            masks.append(sample_holder.get_features_with_parameters(params, feature_type))
        return masks

def sample_from_feature_extractor(feature_extractor: FeatureExtracor, num_samples: int) -> SampleHolder:
    """
    Generate a single SampleHolder containing num_samples samples from the given
    FeatureExtractor

    Parameters
    ----------
    feature_extractor: FeatureExtractor
        The FeatureExtractor (image) to sample from
    num_samples: int
        The number of samples to draw
    """
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
