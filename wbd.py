import json
from wbd import board_calibration, board_transform, postprocessing
import cv2 as cv
import numpy as np
import argparse
import requests

ap = argparse.ArgumentParser()
ap.add_argument('--image-path', help='Grab image from specified file')
ap.add_argument('--image-url', help='Grab image from specified URL')
ap.add_argument('--image-rtsp', help='Grab image from specified RTSP stream')
ap.add_argument('--calibrate', help='Calibrate board points using GUI and save to specified file', default=False)
ap.add_argument('--transform', help='Transform board using points from specified file(s)', nargs='*')
ap.add_argument('--output', help='Save transformed boards to specified file(s)', nargs='*')
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
        exit(1)
    original = frame
else:
    print("Neither `--image-path`, `--image-url` or `--image-rtsp` was specified, use `--help` to print usage")
    exit(1)

if args["calibrate"]:
    h, w = original.shape[:2]
    w_ = 1080
    h_ = 720
    k_w = w/w_
    k_h = h/h_
    preview = cv.resize(original, (w_, h_))
    points = board_calibration.board_calibration(preview)

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
elif args["transform"]:
    idx = 0
    for transform in args["transform"]:
        with open(transform) as f:
            data = json.load(f)
            points = np.array(data["points"], dtype="float32")
            result = board_transform.four_point_transform(original, points, data["aspectRatio"])
            result = postprocessing.postprocessing(result, data["brightness"], data["contrast"])

            if args["output"] and len(args["output"]) > 0:
                cv.imwrite(args["output"].pop(0), result)

        idx = idx + 1

if args["output_original"]:
    if args["image_rtsp"]:
        res, frame = cv.VideoCapture(args["image_rtsp"]).read()
        cv.imwrite(args["output_original"], frame)
