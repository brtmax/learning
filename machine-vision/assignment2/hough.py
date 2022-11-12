import numpy as np
import skimage.io
import os
from typing import Tuple
import copy
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

gray_img = skimage.io.imread("sources/triangle.png")
plt.imshow(gray_img, cmap="gray")
plt.show()

def hough_transofmr_tc(edge_image: np.ndarray, ps: np.ndarray, e=0.5) -> np.ndarray:
    t = ps[0]
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    r = ps[1]
    
    acc = np.zeros_like(t)
    
    edge_pixel_idx = np.where(edge_image > 0)
    for idx in range(len(edge_pixel_idx[0])):
        x = edge_pixel_idx[1][idx]
        y = edge_pixel_idx[0][idx]
        
        res = x * sin_t - y * cos_t + r
        lines_on_point_idx = np.where(np.logical_and(res > -e, res < e))
        acc[lines_on_point_idx] += 1
    return acc

