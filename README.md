# wbd

Whiteboard detection experiments.

Crops and corrects perspective of whiteboard from image.

## Sample

**Original**

<img src="readme_photos/original.jpeg" width="500">

**Right board, cropped and perspecrive wrapped**

<img src="readme_photos/out_right.jpg" width="500">

**Right board, after calibration**

<img src="readme_photos/calibrated_right.jpeg" width="500">

**Left board, cropped and perspecrive wrapped**

<img src="readme_photos/out_left.jpg" width="500">

**Left board, after calibration**

<img src="readme_photos/calibrated_left.jpeg" width="500">

**Postprocessing. Right board**

<img src="readme_photos/postprocessing_right.png" width="500">

**Postprocessing. Left board**

<img src="readme_photos/postprocessing_left.png" width="500">

## Local launch

``` bash
python3 wbd.py --image-path readme_photos/original.jpeg --mode right --output right.png
```
