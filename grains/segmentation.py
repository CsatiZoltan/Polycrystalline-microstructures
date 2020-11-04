"""
This module contains the Segmentation class, responsible for the image
segmentation of grain-based materials (rocks, metals, etc.)

Classes
-------
.. autosummary::
    :nosignatures:
    :toctree: classes/

    Segmentation

"""

import os.path as path

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt
from skimage import io, segmentation, color, measure
from skimage.future import graph
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

from .gala_light import imextendedmin


class Segmentation:
    """Segmentation of grain-based microstructures

    Attributes
    ----------
    original_image : ndarray
        Matrix representing the initial, unprocessed image.
    save_location : str
        Directory where the processed images are saved
    """

    def __init__(self, image_location, save_location=None, interactive_mode=True):
        """Initialize the class with file paths and with some options

        Parameters
        ----------
        image_location : str
            Path to the image to be segmented, file extension included.
        save_location : str, optional
            Path to directory where images will be outputted. If not given, the
            same directory is used where the input image is loaded from.
        interactive_mode : bool, optional
            When True, images of each image manipulation step are plotted and
            details are shown in the console.
            Default is False.

        Returns
        -------
        None.
        """

        # Check inputs
        if not path.isfile(image_location):
            raise Exception('Image file {0} does not exist'.format(image_location))
        extension = path.splitext(image_location)[1][1:]
        allowed_extensions = ['png', 'bmp', 'tiff']
        if extension.lower() not in allowed_extensions:
            raise Exception('Unsupported image file type {0}. Choose from one of the following \
             image types: {1}.'.format(extension, allowed_extensions))
        self.image_location = image_location
        if save_location is None:
            self.save_location = path.dirname(image_location)
        else:
            self.save_location = save_location
        self.__interactive_mode = interactive_mode
        self.__stored_graph = None
        # Load the image and optionally show it
        self.original_image = io.imread(image_location)
        if self.__interactive_mode:
            io.imshow(self.original_image)
            io.show()
            print('Image successfully loaded.')

    def filter_image(self, window_size, image_matrix=None):
        """Median filtering on an image.
        The median filter is useful in our case as it preserves the important
        borders (i.e. the grain boundaries).

        Parameters
        ----------
        window_size : int
            Size of the sampling window.
        image_matrix : 3D ndarray with size 3 in the third dimension, optional
            Input image to be filtered. If not given, the original image is used.

        Returns
        -------
        filtered_image : 3D ndarray with size 3 in the third dimension
            Filtered image, output of the median filter algorithm.
        """

        if image_matrix is None:
            image = self.original_image
        else:
            not_ndarray = not isinstance(image_matrix, np.ndarray)
            wrong_shape = len(np.shape(image_matrix)) != 3 or np.shape(image_matrix)[2] != 3
            if not_ndarray or wrong_shape:
                raise Exception('3D ndarray with size 3 in the third dimension expected.')
            image = image_matrix
        filtered_image = ndi.median_filter(image, window_size)
        if self.__interactive_mode:
            io.imshow(filtered_image)
            io.show()
            print('Median filtering finished.')
        return filtered_image

    def initial_segmentation(self, *args):
        """Perform the quick shift superpixel segmentation on an image.
        The quick shift algorithm is invoked with its default parameters.

        Parameters
        ----------
        *args : 3D numpy array with size 3 in the third dimension
            Input image to be segmented. If not given, the original image is used.

        Returns
        -------
        segment_mask : ndarray
            Label image, output of the quick shift algorithm.
        """

        image = args[0] if args else self.original_image
        segment_mask = segmentation.quickshift(image)
        if self.__interactive_mode:
            io.imshow(color.label2rgb(segment_mask, self.original_image, kind='avg'))
            io.show()
            print('Quick shift segmentation finished. '
                  'Number of segments: {0}'.format(np.amax(segment_mask)))
        return segment_mask

    def merge_clusters(self, segmented_image, threshold=5):
        """Merge tiny superpixel clusters.
        Superpixel segmentations result in oversegmented images. Based on graph
        theoretic tools, similar clusters are merged.

        Parameters
        ----------
        segmented_image : ndarray
            Label image, output of a segmentation.
        threshold : float, optional
            Regions connected by edges with smaller weights are combined.

        Returns
        -------
        merged_superpixels : ndarray
            The new labelled array.
        """

        if self.__stored_graph is None:
            # Region Adjacency Graph (RAG) not yet determined -> compute it
            g = graph.rag_mean_color(self.original_image, segmented_image)
            self.__stored_graph = g
        else:
            g = self.__stored_graph
        merged_superpixels = graph.cut_threshold(segmented_image, g, threshold, in_place=False)
        if self.__interactive_mode:
            io.imshow(color.label2rgb(merged_superpixels, self.original_image, kind='avg'))
            io.show()
            print('Tiny clusters merged. '
                  'Number of segments: {0}'.format(np.amax(merged_superpixels)))
        return merged_superpixels

    def find_grain_boundaries(self, segmented_image):
        """Find the grain boundaries.

        Parameters
        ----------
        segmented_image : ndarray
            Label image, output of a segmentation.

        Returns
        -------
        boundary : bool ndarray
            A bool ndarray, where True represents a boundary pixel.
        """

        boundary = segmentation.find_boundaries(segmented_image)
        if self.__interactive_mode:
            # Superimpose the boundaries of the segmented image on the original image
            superimposed = segmentation.mark_boundaries(self.original_image,
                                                        segmented_image, mode='thick')
            io.imshow(superimposed)
            io.show()
            print('Grain boundaries found.')
        return boundary

    def create_skeleton(self, boundary_image):
        """Use thinning on the grain boundary image to obtain a single-pixel wide skeleton.

        Parameters
        ----------
        boundary_image : bool ndarray
            A binary image containing the objects to be skeletonized.

        Returns
        -------
        skeleton : bool ndarray
            Thinned image.
        """

        skeleton = skeletonize(boundary_image)
        if self.__interactive_mode:
            io.imshow(skeleton)
            io.show()
            print('Skeleton constructed.')
        return skeleton

    def watershed_segmentation(self, skeleton):
        """Watershed segmentation of a granular microstructure.
        Uses the watershed transform to label non-overlapping grains in a cellular
        microstructure given by the grain boundaries.

        Parameters
        ----------
        skeleton : bool ndarray
            A binary image, the skeletonized grain boundaries.

        Returns
        -------
        segmented : ndarray
            Label image, output of the watershed segmentation.
        """
        if skeleton.dtype.name != 'bool':
            raise Exception('A numpy array of type bool expected.')
        # Create a distance function whose maxima will serve as watershed basins
        distance_function = distance_transform_edt(1 - skeleton)
        # Turn the distance function to a negative distance function for watershed
        distance_function = np.negative(distance_function)
        # Do not yet use watershed as that would result an oversegmented image
        # (each local minima of the distance function would become a catchment basin).
        # Hence, first execute the extended-minima transform to find the regional minima
        mask = imextendedmin(distance_function, 2)
        # The watershed segmentation can now be performed
        labelled = measure.label(mask)
        segmented = watershed(distance_function, labelled)
        if self.__interactive_mode:
            io.imshow(color.label2rgb(segmented))
            io.show()
            print('Watershed segmentation finished. '
                  'Number of segments: {0}'.format(np.amax(segmented)))
        return segmented

    def save_image(self, filename, array, is_label_image=False):
        """Save an image as a numpy array.
        The array is saved in the standard numpy format, into the directory determined by the
        `save_location` attribute.

        Parameters
        ----------
        filename : str
            The array is saved under this name, with extension .npy
        array : ndarray
            An image represented as a numpy array.
        is_label_image : bool
            True if the array represents a label image

        """
        # local_vars = list(locals().items())
        # for key, val in local_vars:
        #    if type(val) is np.ndarray and (val == array).all():
        #        break
        name = path.join(self.save_location, filename)
        if is_label_image:
            nlabel = len(np.unique(array))
            io.imsave(name, color.label2rgb(array, colors=np.random.random((nlabel, 3))))
        else:
            io.imsave(name, array)

    def save_array(self, filename, array):
        """Save an image as a numpy array.
        The array is saved in the standard numpy format, into the directory determined by the
        `save_location` attribute.

        Parameters
        ----------
        filename : str
            The array is saved under this name, with extension .npy

        array : ndarray
            An image represented as a numpy array.

        """
        np.save(path.join(self.save_location, filename), array)
