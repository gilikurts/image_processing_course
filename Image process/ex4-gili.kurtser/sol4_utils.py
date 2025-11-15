from scipy.signal import convolve2d
import numpy as np
import imageio
from skimage.color import rgb2gray
from scipy import ndimage
import os

GRAYSCALE = 1
RGB = 2


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def is_rgb(img):
    """
    check if img is RGB or grayscale
    :param img:we assume that the input is only gryscale or rgb image
    :return: True is RGB false else
    """
    dim = len(img.shape)
    if dim == 3:
        return True
    elif dim == 2:
        return False


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as a np.float64 matrix normalized to [0,1]
    """
    image = imageio.imread(filename).astype(np.float64)
    norm_image = image / int(image.max())
    if is_rgb(norm_image):
        if representation == RGB:
            return norm_image
        if representation == GRAYSCALE:
            return rgb2gray(norm_image)
    else:
        if representation == GRAYSCALE:
            return norm_image
        else:
            return None
    return None

def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    im = ndimage.convolve(im, np.transpose(blur_filter), mode='mirror')
    im = ndimage.convolve(im, blur_filter, mode='mirror')
    return im[::2, ::2]


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    new_image = np.zeros(2 * np.array(im.shape))
    new_image[::2, ::2] = im
    new_image = ndimage.convolve(new_image, 2 * np.transpose(blur_filter), mode='mirror')
    new_image = ndimage.convolve(new_image, 2 * blur_filter, mode='mirror')
    return new_image


def create_filter_vec(filter_size):
    down_function = np.array([1, 1])
    result_con = down_function
    for i in range(1, filter_size - 1):
        result_con = np.convolve(result_con, down_function)
    filter_vec = (np.array([result_con]) / np.sum(result_con)).astype('float64')
    return filter_vec


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    filter_vec = create_filter_vec(filter_size)
    pyr = []
    pyr.append(im)
    for level in range(1, max_levels):
        new_img = pyr[-1]
        if min(new_img.shape[0], new_img.shape[1]) < 32:
            break
        else:
            reduced_img = reduce(new_img, filter_vec)
            pyr.append(reduced_img)
    return pyr, filter_vec