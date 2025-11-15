import numpy as np
import imageio
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, rgb2grey, yiq2rgb, rgb2yiq, gray2rgb
from scipy import ndimage
import os

GRAYSCALE = 1
RGB = 2
MIN_FIXEL_SIZE = 32


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
        if min(new_img.shape[0], new_img.shape[1]) < MIN_FIXEL_SIZE:
            break
        else:
            reduced_img = reduce(new_img, filter_vec)
            pyr.append(reduced_img)
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
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
    pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    lpyr = []
    for i in range(0, len(pyr) - 1):
        new_level = (pyr[i] - expand(pyr[i + 1], filter_vec))
        lpyr.append(new_level)
    lpyr.append(pyr[len(pyr) - 1])
    return lpyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    image = lpyr[-1]
    for level in range(len(coeff) - 2, - 1, - 1):
        image_i = coeff[level] * lpyr[level] + expand(image, filter_vec)
        image = image_i
    return image_i


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    width = 0
    for image in range(0, min(levels, len(pyr))):
        width += pyr[image].shape[1]
    black_img_dim = (pyr[0].shape[0], width)
    res = np.zeros(black_img_dim)

    cur_place = 0
    for i in range(0, min(levels, len(pyr))):
        height, width = pyr[i].shape
        res[0: height, cur_place: cur_place + width] = \
            (pyr[i] - np.min(pyr[i])) / (np.max(pyr[i]) - np.min(pyr[i]))
        cur_place += width
    img = res
    return img


def display_pyramid(pyr, levels):
    """
    display the rendered pyramid
    """
    res = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(res, cmap=plt.cm.gray)
    plt.show()
    return None


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    L1, filter_vec_im = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2, filter_vec_im = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    mask_pyr, filter_vec_mask = build_gaussian_pyramid(mask.astype('float64'), max_levels, filter_size_mask)
    L1, L2, mask_pyr = np.array(L1), np.array(L2), np.array(mask_pyr)
    coeff = np.ones(len(mask_pyr)).tolist()

    blend = mask_pyr * L1 + (np.ones(mask_pyr.shape) - mask_pyr) * L2
    blend = laplacian_to_image(blend, filter_vec_im, coeff)
    blend = (blend - np.min(blend)) / (np.max(blend) - np.min(blend))
    return blend


def relpath(filename):
    """
    help function from os.
    :param filename: file's path.
    :return: relative path.
    """
    return os.path.join(os.path.dirname(__file__), filename)


def blending_example2():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """

    filter_size = 5
    max_level = 8
    mona_lisa = read_image(relpath('external/MONAIMG2.jpg'), 2)
    avital = read_image(relpath('external/avital_img2.jpg'), 2)
    mask = read_image(relpath('external/mask2.jpg'), 1).astype(int)
    blend_img = np.empty_like(mona_lisa)
    blend_img[:, :, 0] = pyramid_blending(avital[:, :, 0], mona_lisa[:, :, 0],
                                          mask, max_level, filter_size,
                                          filter_size)
    blend_img[:, :, 1] = pyramid_blending(avital[:, :, 1], mona_lisa[:, :, 1],
                                          mask, max_level, filter_size,
                                          filter_size)
    blend_img[:, :, 2] = pyramid_blending(avital[:, :, 2], mona_lisa[:, :, 2],
                                          mask, max_level, filter_size,
                                          filter_size)
    fig = plt.figure()
    sub_figure = fig.add_subplot(2, 2, 1)
    plt.imshow(mona_lisa)
    sub_figure = fig.add_subplot(2, 2, 2)
    plt.imshow(avital)
    sub_figure = fig.add_subplot(2, 2, 3)
    plt.imshow(mask, cmap=plt.cm.gray)
    sub_figure = fig.add_subplot(2, 2, 4)
    plt.imshow(blend_img)
    plt.show()
    return mona_lisa, avital, mask.astype(bool), blend_img


def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    filter_size = 7
    max_level = 8
    red_and_kitty = read_image(relpath('external/redkitty_img1.jpg'), 2)
    shmuel = read_image(relpath('external/shmuel2.jpg'), 2)
    mask = read_image(relpath('external/mask1.jpg'), 1).astype(int)
    blend_img = np.empty_like(red_and_kitty)
    blend_img[:, :, 0] = pyramid_blending(shmuel[:, :, 0], red_and_kitty[:, :, 0],
                                          mask, max_level, filter_size,
                                          filter_size)
    blend_img[:, :, 1] = pyramid_blending(shmuel[:, :, 1], red_and_kitty[:, :, 1],
                                          mask, max_level, filter_size,
                                          filter_size)
    blend_img[:, :, 2] = pyramid_blending(shmuel[:, :, 2], red_and_kitty[:, :, 2],
                                          mask, max_level, filter_size,
                                          filter_size)
    fig = plt.figure()
    sub_figure = fig.add_subplot(2, 2, 1)
    plt.imshow(red_and_kitty)
    sub_figure = fig.add_subplot(2, 2, 2)
    plt.imshow(shmuel)
    sub_figure = fig.add_subplot(2, 2, 3)
    plt.imshow(mask, cmap=plt.cm.gray)
    sub_figure = fig.add_subplot(2, 2, 4)
    plt.imshow(blend_img)
    plt.show()
    return red_and_kitty, shmuel, mask.astype(bool), blend_img


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


