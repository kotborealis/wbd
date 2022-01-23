"""
This module provides functions for computer vision.
"""

import sys
from typing import Union, List, Tuple, Optional

import numpy as np
from PIL import Image


def rgb_points_scatter(points: np.ndarray):
    """
    Represents colors from rgb triples in points on scatter graph.

    :param points: array of a 24-bit RGB triples
    """
    import matplotlib.pyplot as plt

    x = np.random.rand(1000)
    fig, ax = plt.subplots()

    for ind, point in enumerate(points):
        ax.scatter(x + ind, np.random.gamma(1, size=1000), c=[[el / 255 for el in point]])

    ax.set_facecolor('black')
    ax.set_title('Points colors')

    fig.set_figwidth(8)
    fig.set_figheight(8)

    plt.show()


def pack_rgb(rgb: Union[np.ndarray, Tuple[int], List[int]]) -> np.ndarray:
    """
    Packs a 24-bit RGB triples into a single integer.
    Conversion saves original shape.

    WORKS ONLY with RGB/BGR with 3 color layouts!

    Example:
        np.array([
            [[1, 1, 1],
            [2, 2, 2],
            [6, 6, 6],
            ],
            [[3, 3, 3],
            [4, 4, 4],
            [7, 7, 7]]
        ])
        converts to np.array([
            [ 65793, 131586, 394758],
            [197379, 263172, 460551]])
    """
    original_shape = None

    if isinstance(rgb, np.ndarray):
        # Image case - numpy array of pixels
        assert rgb.shape[-1] == 3  # check layouts count
        original_shape = rgb.shape[:-1]  # except last one with rgb values
    else:
        # Pixel case
        assert len(rgb) == 3
        rgb = np.array(rgb)

    # 4x3x3 becomes 9x4 matrix
    rgb = rgb.astype(int).reshape((-1, 3))

    packed: np.ndarray = (rgb[:, 0] << 16 | rgb[:, 1] << 8 | rgb[:, 2])

    if original_shape is not None:
        return packed.reshape(original_shape)
    return packed


def unpack_rgb(packed_rgb: Union[np.ndarray, int]) -> np.ndarray:
    """
    Unpacks a single integer or array of integers into one or more
    24-bit RGB values.

    Example:
        np.array([
            [ 65793, 131586, 394758],
            [197379, 263172, 460551]])
        converts to
        np.array([
            [[1, 1, 1],
            [2, 2, 2],
            [6, 6, 6],
            ],
            [[3, 3, 3],
            [4, 4, 4],
            [7, 7, 7]]
        ])
    """
    original_shape = None

    if isinstance(packed_rgb, np.ndarray):
        assert packed_rgb.dtype == int
        original_shape = packed_rgb.shape

        packed_rgb = packed_rgb.reshape(-1, 1)

    rgb = ((packed_rgb >> 16) & 0xff,
           (packed_rgb >> 8) & 0xff,
           packed_rgb & 0xff)

    if original_shape is None:
        # Single mod - unpacks a single integer
        return np.array(rgb)
    return np.hstack(rgb).reshape((*original_shape, 3))


def load(input_filename: str) -> Optional[np.ndarray]:
    """Loads and converts image from input_filename to numpy array"""
    try:
        pil_img = Image.open(input_filename)
    except IOError:
        sys.stderr.write('warning: error opening {}\n'.format(
            input_filename))
        return None

    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    return np.array(pil_img)
