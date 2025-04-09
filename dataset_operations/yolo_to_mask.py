import numpy as np

from ultralytics.data.utils import polygon2mask

imgsz = (1080, 810)
polygon = np.array([805, 392, 797, 400, ..., 808, 714, 808, 392])  # (238, 2)

mask = polygon2mask(
    imgsz,  # tuple
    [polygon],  # input as list
    color=255,  # 8-bit binary
    downsample_ratio=1,
)