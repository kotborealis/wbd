"""
Image postprocessing.

apply_brightness_contrast: https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
"""
import cv2

from numpy import ndarray
from typing import Union


def apply_brightness_contrast(input_img: ndarray, brightness: Union[int, float] = 0, contrast: Union[int, float] = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def postprocessing(output_path: str, crop_weights: list):
    img = cv2.imread(output_path)

    # crop_weights[0] - up/down coefficients.
    # crop_weights[1] - left/right coefficients.
    original_shape: tuple = img.shape
    crop_img = img[crop_weights[0][0]:original_shape[0] + crop_weights[0][1],
               crop_weights[1][0]:original_shape[1] + crop_weights[1][1]]

    _back_indx: int = output_path.rfind('/') + 1
    cv2.imwrite(output_path[:_back_indx] + 'postprocessing_' + output_path[_back_indx:], crop_img)
