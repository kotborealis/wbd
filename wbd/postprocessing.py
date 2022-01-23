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


# def postprocessing(input, brightness=0, contrast=0):
#     return apply_brightness_contrast(input, brightness, contrast)
