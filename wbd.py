import os
import sys
import json
import argparse

from typing import Optional
from pathlib import Path

import requests
import cv2 as cv
import numpy as np

from wbd import board_calibration, four_point_transform, \
    apply_brightness_contrast, postprocessing, UnsupportedBoardMode
from calibration import undistort_img

TMP_DIR = os.path.join('.', 'local')

ap = argparse.ArgumentParser()
ap.add_argument('--image-path', help='Grab image from specified file')
ap.add_argument('--image-url', help='Grab image from specified URL')
ap.add_argument('--image-rtsp', help='Grab image from specified RTSP stream')
ap.add_argument('--calibrate', help='Calibrate board points using GUI and save to specified file', default=False)
# ap.add_argument('--transform', help='Transform board using points from specified file(s)', nargs='*')
ap.add_argument('--mode',
                help="Get the side of the board by passing one of the operating modes: ('left', 'right', 'sheet')",
                nargs='*')
ap.add_argument('--postprocessing', help="Execute image postprocessing", action='store_true')
ap.add_argument('--output', help='Save transformed boards to specified file(s)', nargs='*')
# ap.add_argument('--output-postprocessing', help='Save processed board to specified file')
ap.add_argument('--output-original', help='Save original to specified file(s)')
args = vars(ap.parse_args())

original = []

if args["image_path"]:
    original = cv.imread(args["image_path"])
elif args["image_url"]:
    res = requests.get(args["image_url"], stream=True).content
    original = np.asarray(bytearray(res), dtype="uint8")
    original = cv.imdecode(original, cv.IMREAD_COLOR)
elif args["image_rtsp"]:
    res, frame = cv.VideoCapture(args["image_rtsp"]).read()
    if not res:
        print("Failed to grab frame from specified rtsp stream")
        sys.exit(1)
    original = frame
else:
    print("Neither `--image-path`, `--image-url` or `--image-rtsp` was specified, use `--help` to print usage")
    sys.exit(1)

if args["calibrate"]:
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
        print("Expected 4 points, got " + str(len(points)))
# elif args["transform"]:
#     idx = 0
#     for transform in args["transform"]:
#         with open(transform) as f:
#             data = json.load(f)
#             points = np.array(data["points"], dtype="float32")
#             result = board_transform.four_point_transform(original, points, data["aspectRatio"])
#             result = postprocessing.postprocessing(result, data["brightness"], data["contrast"])
#
#             if args["output"] and len(args["output"]) > 0:
#                 cv.imwrite(args["output"].pop(0), result)
#
#         idx = idx + 1
elif args["mode"]:
    idx = 0
    for mode in args["mode"]:
        coefficients_filename: Optional[str] = None
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
                                          mode=mode.lower())
            result = apply_brightness_contrast(result, data["brightness"], data["contrast"])

            if args["output"] and len(args["output"]) > 0:
                if mode != 'sheet':
                    from time import time

                    output_path = args["output"].pop(0)

                    Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
                    tmp_filename = os.path.join(TMP_DIR, f'tmp_{str(int(time()))}.png')
                    cv.imwrite(tmp_filename, result)
                    undistort_img(filename=tmp_filename, mode=mode.lower(), output_path=output_path)

                    if args['postprocessing']:
                        postprocessing(output_path=output_path, crop_weights=data["сrop_weights"])
                else:
                    cv.imwrite(args["output"].pop(0), result)

        idx = idx + 1

if args["output_original"]:
    if args["image_rtsp"]:
        res, frame = cv.VideoCapture(args["image_rtsp"]).read()
        cv.imwrite(args["output_original"], frame)
