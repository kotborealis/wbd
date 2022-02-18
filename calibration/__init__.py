import logging
from typing import Optional
import cv2
import numpy

from .exceptions import UnsupportedCalibrationMode


def undistort_img(image: numpy.ndarray, output_path: str, mode: str, logger: logging.Logger):
    """
    Undistort image using weights from yaml file.

    :param image: image for calibration
    :param output_path: where to save the file after processing
    :param mode: board side
    :param logger: logger
    """
    logger.info("Калибровка изображения")
    from .calibration import get_calibration_weights

    img_dir: Optional[str] = None
    if mode == 'left':
        img_dir: str = 'left_board_calibration'
    elif mode == 'right':
        img_dir: str = 'right_board_calibration'

    logger.info(f"Выбран режим калибровки: {mode}. img_dir: {img_dir}")
    if img_dir is None:
        raise UnsupportedCalibrationMode

    mtx, dist = get_calibration_weights(img_dir=img_dir, mode=mode)

    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    cv2.imwrite(output_path, dst)
    logger.info(f"Основной файл готов и записан по адресу: {output_path}")
