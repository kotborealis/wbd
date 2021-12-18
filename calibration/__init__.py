from .exceptions import UnsupportedCalibrationMode


def undistort_img(filename: str, output_path: str, mode: str):
    """
    Undistort image using weights from yaml file.

    :param filename: path to the source file
    :param output_path: where to save the file after processing(transmitted without output filename, only path)
    :param mode: board side
    """
    from typing import Optional
    import cv2
    from .calibration import get_calibration_weights

    img_dir: Optional[str] = None
    if mode == 'left':
        img_dir: str = 'left_board_calibration'
    elif mode == 'right':
        img_dir: str = 'right_board_calibration'

    if img_dir is None:
        raise UnsupportedCalibrationMode

    mtx, dist = get_calibration_weights(img_dir=img_dir, mode=mode)

    original = cv2.imread(filename)
    h, w = original.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(original, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    # cv2.imwrite('calibresult.png', dst)
    # убрать /
    output_file_path: str = output_path + filename[filename.rfind('/') + 1:]
    cv2.imwrite(output_file_path, dst)
