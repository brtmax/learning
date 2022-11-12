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

def hough_transform_tc(edge_image: np.ndarray, ps: np.ndarray, e=0.5) -> np.ndarray:
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

def span_tc_parameter_space(*, c_max: float, shape=(200, 200)) -> np.ndarray:
    # Span values of theta over the desired range (-90.0 -> 90.0 degree in rad)
    theta = np.arange(np.deg2rad(-90),np.deg2rad(90))
    # Span values of c over the desired range
    c = np.arange(0, c_max)
    return np.meshgrid(theta, c)

# span parameter space and calcualte accumulator
t_c = span_tc_parameter_space(c_max=100, shape=(200,200))
acc_tc = hough_transform_tc(edge_image=gray_img, ps=t_c)

# visualize
fig, ax = plt.subplots()
ax.imshow(acc_tc, cmap='gray')
ax.set_yticks(np.linspace(0, acc_tc.shape[1]-1, 5))
ax.set_yticklabels([-100,-50,0,50,100])
ax.set_ylabel("c")
ax.set_xticks(np.linspace(0, acc_tc.shape[1], 5))
ax.set_xticklabels(np.round(np.deg2rad([-90,-45,0,45,90]), 2))
ax.set_xlabel("t")
plt.show()