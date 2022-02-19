import os
import sys
import json
import argparse

from logging import Logger
from typing import Optional
from pathlib import Path

import requests
import cv2 as cv
import numpy as np

from calibration import undistort_img
from wbdlogger import WBDLogger

TMP_DIR = os.path.join('.', 'local')
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument('--image-path', help='Grab image from specified file')
ap.add_argument('--image-url', help='Grab image from specified URL')
ap.add_argument('--image-rtsp', help='Grab image from specified RTSP stream')
ap.add_argument('--calibrate', help='Calibrate board points using GUI and save to specified file', default=False)
ap.add_argument('--mode',
                help="Get the side of the board by passing one of the operating modes: ('left', 'right', 'sheet')",
                nargs='*')
ap.add_argument('--output', help='Save transformed boards to specified file(s)', nargs='*')
ap.add_argument('--output-original', help='Save original to specified file(s)')
args = vars(ap.parse_args())

original = []
ROOT_LOGGER: Logger = WBDLogger()

if args["image_path"]:
    original = cv.imread(args["image_path"])
elif args["image_url"]:
    res = requests.get(args["image_url"], stream=True).content
    original = np.asarray(bytearray(res), dtype="uint8")
    original = cv.imdecode(original, cv.IMREAD_COLOR)
elif args["image_rtsp"]:
    res, frame = cv.VideoCapture(args["image_rtsp"]).read()
    if not res:
        ROOT_LOGGER.info("Failed to grab frame from specified rtsp stream")
        sys.exit(1)
    original = frame
else:
    ROOT_LOGGER.info(
        "Neither `--image-path`, `--image-url` or `--image-rtsp` was specified, use `--help` to print usage")
    sys.exit(1)

if args["calibrate"]:
    from wbd import board_calibration
    h, w = original.shape[:2]
    w_ = 1080
    h_ = 720
    k_w = w / w_
    k_h = h / h_
    preview = cv.resize(original, (w_, h_))
    points = board_calibration(preview)

    if len(points) == 4:
        points = [(p_x * k_w, p_y * k_h) for p_x, p_y in points]
        with open(args["calibrate"], 'w') as f:
            json.dump({
                "points": points,
                "aspectRatio": 1.5,
                "brightness": 0,
                "contrast": 0
            }, f)
    else:
        ROOT_LOGGER.info("Expected 4 points, got " + str(len(points)))

elif args["mode"]:
    from wbd import postprocessing, four_point_transform, \
        apply_brightness_contrast, unsharp_mask, UnsupportedBoardMode

    idx = 0
    for mode in args["mode"]:
        coefficients_filename: Optional[str] = None

        ROOT_LOGGER.info(f"Mode: {mode}")

        if mode == 'right':
            coefficients_filename = './board_right.json'
        if mode == 'left':
            coefficients_filename = './board_left.json'
        if mode == 'sheet':
            coefficients_filename = './sheet.json'

        if coefficients_filename is None:
            raise UnsupportedBoardMode

        with open(coefficients_filename) as f:
            data = json.load(f)
            points = np.array(data["points"], dtype="float32")

            result = four_point_transform(image=original, pts=points,
                                          aspectRatio=data["aspectRatio"],
                                          mode=mode.lower(),
                                          logger=ROOT_LOGGER)

            result = unsharp_mask(image=result)
            result = apply_brightness_contrast(result, data["brightness"], data["contrast"])

            if args["output"] and len(args["output"]) > 0:
                if mode != 'sheet':
                    output_path = args["output"].pop(0)
                    undistort_img(image=result, mode=mode.lower(), output_path=output_path, logger=ROOT_LOGGER)
                    postprocessing(output_path=output_path, crop_weights=data["сrop_weights"], tmp_dir=TMP_DIR)
                else:
                    cv.imwrite(args["output"].pop(0), result)

        idx = idx + 1

if args["output_original"]:
    if args["image_rtsp"]:
        res, frame = cv.VideoCapture(args["image_rtsp"]).read()
        cv.imwrite(args["output_original"], frame)
