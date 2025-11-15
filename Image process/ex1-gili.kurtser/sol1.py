import numpy as np
import matplotlib.pyplot as plt
import scipy
import imageio
from skimage.color import rgb2gray
import imageio

GRAYSCALE = 1
RGB = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])
NUMBER_OF_BINS = 256
LAST = -1


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


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    image = read_image(filename, representation)
    plt.figure()
    if representation == RGB:
        plt.imshow(image)
    elif representation == GRAYSCALE:
        plt.imshow(image, cmap=plt.cm.gray)
    else:
        return None
    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    yiq_image = np.empty_like(imRGB)
    red, green, blue = imRGB[:, :, 0], imRGB[:, :, 1], imRGB[:, :, 2]
    Y = np.dot(0.299, red) + np.dot(0.587, green) + np.dot(0.114, blue)
    I = np.dot(0.596, red) + np.dot(-0.275, green) + np.dot(-0.321,
                                                            blue)
    Q = np.dot(0.212, red) + np.dot(-0.523, green) + np.dot(0.311, blue)
    yiq_image[:, :, 0], yiq_image[:, :, 1], yiq_image[:, :, 2] = Y, I, Q

    return yiq_image


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    imRGB = np.empty_like(imYIQ)
    Y, I, Q = imYIQ[:, :, 0], imYIQ[:, :, 1], imYIQ[:, :, 2]
    red = np.dot(1, Y) + np.dot(0.956, I) + np.dot(0.619, Q)
    green = np.dot(1, Y) + np.dot(-0.272, I) + np.dot(-0.647,
                                                      Q)
    blue = np.dot(1, Y) + np.dot(-1.106, I) + np.dot(1.703, Q)
    imRGB[:, :, 0], imRGB[:, :, 1], imRGB[:, :, 2] = red, green, blue

    return imRGB


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """
    if not is_rgb(im_orig):
        im_eq, hist_orig, hist_eq = make_equalize(im_orig)
        return [im_eq, hist_orig, hist_eq]
    if is_rgb(im_orig):
        new_image = rgb2yiq(im_orig)
        Y = new_image[:, :, 0]
        im_eq_tmp, hist_orig, hist_eq = make_equalize(Y)
        new_image[:, :, 0] = im_eq_tmp
        im_eq = yiq2rgb(new_image)
        return [im_eq, hist_orig, hist_eq]


def make_equalize(img):
    """
    I used the idea from this page - https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
    :param img:
    :return: the parameters for  histogram_equalize
    """
    img = np.round(img * 255).astype('int')
    hist_orig, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist_orig.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    if cdf_m.max() - cdf_m.min() == 0:
        return img
    else:
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        im_eq = cdf[img]
        hist_eq, bins_eq = np.histogram(im_eq.flatten(), 256, [0, 256])
        im_eq = im_eq.astype(np.float64) / 255
        return im_eq, hist_orig, hist_eq


def computing_z(q):
    """
    calculate z component
    :param n_quant:
    :return:
    """
    z = np.zeros((len(q) + 1,), dtype=int)
    z[-1] = 255
    z[0] = -1
    for i in range(1, len(z) - 1):
        z[i] = np.floor([(q[i - 1] + q[i]) / 2]).astype(int)
    return z


def computing_q(z, histogram):
    """
    calculate q component
    :param z:
    :param histogram:
    :return: q
    """
    q = np.zeros((len(z) - 1,), dtype=int)
    for i in range(len(q)):
        lower_part = 0
        upper_part = 0
        for g in range(z[i] + 1, z[i + 1]):
            lower_part += histogram[g]
            upper_part += g * histogram[g]
        q[i] = np.round(upper_part / lower_part)
    return q


def calculate_error(z, q, histogram):
    """
    calculate the error rate
    :param z:
    :param q:
    :param histogram:
    :return:
    """
    calc_error = 0
    for i in range(len(q)):
        calc_error += (np.square(q[i] - np.array(range(z[i] + 1, z[i + 1] + 1)))).dot(
            (histogram[z[i] + 1: z[i + 1] + 1]))
    return calc_error


def find_initial_z(n_quant, histogram):
    """

    :param n_quant:
    :param histogram:
    :return:
    """
    pixels_num = np.cumsum(histogram)[-1]
    cum_hist = np.cumsum(histogram)
    pixels_for_segment = np.round(pixels_num / n_quant)
    z = np.zeros((n_quant + 1,), dtype=int)
    z[-1] = 255
    for i in range(1, n_quant):
        z[i] = np.round(np.where(cum_hist >= (pixels_for_segment * i))[0][0])
    return z


def make_quantization(img, n_quant, n_iter):
    """
    :param img:
    :param n_quant:
    :param n_iter:
    :return:
    """
    histogram, bins = np.histogram(img, 256)
    histogram = histogram/max(histogram)
    z = find_initial_z(n_quant, histogram)
    error_array = []
    for i in range(0, n_iter):
        q = computing_q(z, histogram)
        new_error = calculate_error(z, q, histogram)
        error_array.append(new_error)
        img = calculate_quantified_img(img, z, q)
        if np.array_equal(z, computing_z(q)):
            break
        z = computing_z(q)
    return [img, error_array]


def calculate_quantified_img(img, z, q):
    """

    :param img:
    :param z:
    :param q:
    :return:
    """
    cdf = np.zeros(256)
    for i in range(len(z) - 1):
        cdf[z[i]:z[i + 1] + 1] = q[i]
    im_qa = cdf[np.round(img).astype('int')]
    return im_qa


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    note: I used some ideas to the  help functions from here:
    https://stackoverflow.com/questions/28196787/image-quantization-in-matlab
    """
    if not is_rgb(im_orig):
        im_orig = np.round(im_orig * 255)
        im_quant, error = make_quantization(im_orig, n_quant, n_iter)
        return [im_quant, error]
    else:
        im_orig = np.round(im_orig * 255)
        new_image = rgb2yiq(im_orig)
        Y = new_image[:, :, 0]
        im_quant, error = make_quantization(Y, n_quant, n_iter)
        new_image[:, :, 0] = im_quant
        im_quant = yiq2rgb(new_image)
        return [im_quant.astype(np.float64) / 255, error]


def quantize_rgb(im_orig, n_quant):  # Bonus - optional
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input RGB image of type float64 in the range [0,1]
    :param n_quant: Number of intensities im_quant image will have
    :return:  im_quant - the quantized output image
    """
    pass


x = read_image("DSC05404.jpg",1)
print(x.shape)