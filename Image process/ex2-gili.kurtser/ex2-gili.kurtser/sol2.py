import numpy as np
import imageio
from skimage.color import rgb2gray
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.ndimage.interpolation import map_coordinates
from scipy import signal


GRAYSCALE = 1
RGB = 2


def DFT(signal):
    """
    calculate fourier transform of signal (1-D)
    :param signal:
    :return:
    """

    N = len(signal)
    n = np.arange(N)
    u = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * u * n / N)

    fourier_signal = np.dot(e, signal)

    return fourier_signal


def IDFT(fourier_signal):
    """
     calculate oppisite fourier transform of signal (1-D)
    :param fourier_signal:
    :return:
    """
    N = len(fourier_signal)
    n = np.arange(N)
    u = n.reshape((N, 1))
    e = np.exp(2j * np.pi * u * n / N)
    inverse_f_signal = (1 / N) * np.dot(e, fourier_signal)

    return inverse_f_signal


def DFT2(image):
    """
    2D furier transform - start with row and then goes to columns
    :param image:
    :return:
    """
    new_image = np.empty_like(image).astype("complex128")
    for index, row in enumerate(image):
        new_image[index] = DFT(row)
    new_image_t = np.transpose(new_image)
    for index, row in enumerate(new_image_t):
        new_image_t[index] = DFT(row)
    return np.transpose(new_image_t)


def IDFT2(fourier_image):
    """

    :param fourier_image:
    :return:
    """
    new_image = np.empty_like(fourier_image).astype("complex128")
    for index, row in enumerate(fourier_image):
        new_image[index] = IDFT(row)
    new_image_t = np.transpose(new_image)
    for index, row in enumerate(new_image_t):
        new_image_t[index] = IDFT(row)
    return np.transpose(new_image_t)


def change_rate(filename, ratio):
    """

    :param filename:
    :param ratio:
    :return:
    """
    samplerate, data = wavfile.read(filename)
    write("change_rate.wav.", int(samplerate * ratio), data)
    return


def resize(data, ratio):
    """
    :param data:
    :param ratio:
    :return:
    """
    if ratio == 1:
        return data
    data = np.fft.fftshift(DFT(data))
    new_size = int(len(data) / ratio)
    if ratio < 1:
        pad_size = int(new_size - len(data))
        residual = pad_size % 2
        pad = pad_size // 2
        if residual == 0:
            new_data = np.pad(data, (pad, pad), constant_values=(0, 0))
        else:
            new_data = np.pad(data, (pad + 1, pad), constant_values=(0, 0))
    if ratio > 1:
        middle = round(len(data) / 2)
        new_size_middle = round(new_size / 2)
        new_data = data[(middle - new_size_middle):(middle + new_size_middle)]
    return np.real(IDFT(np.fft.ifftshift(new_data)))


def change_samples(filename, ratio):
    """

    :param filename:
    :param ratio:
    :return:
    """
    samplerate, data = wavfile.read(filename)
    resize_sample = resize(data, ratio).astype(np.float64)
    write("change_samples.wav.", samplerate, resize_sample)
    # return resize_sample


def resize_spectrogram(data, ratio):
    """

    :param data:
    :param ratio:
    :return:
    """

    spectrogram = stft(data)
    new_spec = []
    for row in spectrogram:
        new_spec.append(resize(row, ratio))
    new_spec_np = np.array(new_spec)
    return istft(new_spec_np)


def resize_vocoder(data, ratio):
    """

    :param data:
    :param ratio:
    :return:
    """
    return istft(phase_vocoder(stft(data), ratio))


def conv_der(im):
    """

    :param im:
    :return:
    """
    kernel_x = np.array([[0.5, 0, -0.5]])
    kernel_y = kernel_x.T
    dx = signal.convolve2d(im, kernel_x, mode="same")
    dy = signal.convolve2d(im, kernel_y, mode="same")
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


def fourier_der(im):
    """

    :param im:
    :return:
    """
    f_image = np.fft.fftshift(DFT2(im))
    N_x, N_y = f_image.shape
    u = np.arange(-N_x / 2, N_x / 2).reshape((N_x, 1))
    v = np.arange(-N_y / 2, N_y / 2).reshape((1, N_y))
    x_multply = (2 * np.pi * 1j * u)
    y_multply = (2 * np.pi * 1j * v)
    y_for_return = IDFT2(np.fft.ifftshift(f_image * y_multply))
    x_for_return = IDFT2(np.fft.ifftshift(f_image * x_multply))
    magnitude = np.sqrt(np.abs(y_for_return) ** 2 + np.abs(x_for_return) ** 2)
    return magnitude


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


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec

