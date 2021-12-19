"""
https://learnopencv.com/camera-calibration-using-opencv/
https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
https://medium.com/vacatronics/3-ways-to-calibrate-your-camera-using-opencv-and-python-395528a51615
https://github.com/abidrahmank/OpenCV2-Python-Tutorials/blob/master/source/py_tutorials/py_calib3d/py_calibration/py_calibration.rst

CalibrationCoefficients:
camera_matrix - Внутренняя матрица камеры.
dist_coeffs - Коэффициенты искажения объектива.
rotation_vector - Вращение указано как вектор 3×13 × 13×1.
        Направление вектора задает ось вращения, а величина вектора — угол поворота
translation_vector - 3×1 Translation vector(вектор смещения).
"""
import glob
import os
from typing import NamedTuple, NoReturn
from pathlib import Path

import numpy as np
import cv2

CHECKERBOARD: tuple = (6, 9)
CRITERIA: tuple = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
CALIBRATION_DIR: str = os.path.join('.', 'calibration')
DEBUG_CALIBRATION_DIR: str = os.path.join(CALIBRATION_DIR, 'lines_on_chessboard')
WEIGHTS_PATH = os.path.join(CALIBRATION_DIR, 'weights.yml')


class CalibrationCoefficients(NamedTuple):
    """Camera calibration coefficients"""
    ret: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rotation_vector: list
    translation_vector: list


def load_coefficients(mode: str):
    """Loads camera matrix and distortion coefficients."""
    cv_file = cv2.FileStorage(WEIGHTS_PATH, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode(f'K_{mode}').mat()
    dist_matrix = cv_file.getNode(f'D_{mode}').mat()

    cv_file.release()
    return camera_matrix, dist_matrix


def get_calibration_weights(img_dir: str, mode: str) -> NoReturn:
    """Save the camera matrix and the distortion coefficients to given path/file."""

    def update_coefficients_in_weights():
        coefficients: CalibrationCoefficients = _get_calibration_coefficients(img_dir=img_dir)

        cv_file = cv2.FileStorage(WEIGHTS_PATH, cv2.FILE_STORAGE_APPEND)

        cv_file.write(f'K_{mode}', coefficients.camera_matrix)
        cv_file.write(f'D_{mode}', coefficients.dist_coeffs)

        # Closes the file and releases all the memory buffers
        cv_file.release()
        return coefficients.camera_matrix, coefficients.dist_coeffs

    _cm = None
    _dm = None

    if os.path.exists(WEIGHTS_PATH):
        _cm, _dm = load_coefficients(mode)

    if _cm is None or _dm is None:
        # There are no such coefficients or such file
        _cm, _dm = update_coefficients_in_weights()

    return _cm, _dm


def _get_calibration_coefficients(img_dir: str, debug_mode: bool = False) -> CalibrationCoefficients:
    """
    Gets calibration coefficients using test images(from ./calibration directory) with chessboard pattern.

    :param img_dir: name of the directory with images for calibration.
        Dir must be included in ./calibration/
    """
    if debug_mode:
        Path(DEBUG_CALIBRATION_DIR).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(DEBUG_CALIBRATION_DIR, img_dir)).mkdir(parents=True, exist_ok=True)

    # Вектор для хранения векторов трехмерных точек для каждого изображения шахматной доски
    objpoints = []
    # Вектор для хранения векторов 2D точек для каждого изображения шахматной доски
    imgpoints = []

    # Определение мировых координат для 3D точек
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    images = glob.glob(os.path.join(CALIBRATION_DIR, img_dir, '*.jpg'))
    for file in images:
        img = cv2.imread(file)
        img_in_gray_colors = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Поиск углов шахматной доски
        pattern_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        pattern_found, corners = cv2.findChessboardCorners(img_in_gray_colors, CHECKERBOARD, pattern_flags)

        if pattern_found:
            objpoints.append(objp)
            # уточнение координат пикселей для заданных 2d точек.
            corners2 = cv2.cornerSubPix(img_in_gray_colors, corners, (11, 11), (-1, -1), CRITERIA)

            imgpoints.append(corners2)
            if debug_mode:
                _draw_lines_on_chessboard(img=img, corners=corners2, filename=file[file.rfind(img_dir):])

        # Разрешение изображения
        if prev_img_shape is None:
            prev_img_shape = img_in_gray_colors.shape[::-1]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, prev_img_shape,
                                                       None, None)

    return CalibrationCoefficients(ret, mtx, dist, rvecs, tvecs)


def _draw_lines_on_chessboard(img, corners, filename):
    """Using the obtained corners draws lines on a chessboard"""
    img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, True)
    cv2.imwrite(os.path.join(DEBUG_CALIBRATION_DIR, filename), img=img)
