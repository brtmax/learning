import numpy as np
import skimage.io
import os
from typing import Tuple
import copy
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# define the sobel kernel
sobel_kernel_vertical = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
sobel_kernel_horizontal = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])

# Excuse the direct path
gray_img = plt.imread("/Users/maxbretschneider/Desktop/Development/learning/machine-vision/assignment2.py/sources/postit2g.png")

# ensure image is float type
gray_img = gray_img.astype(float)

# implement convolution2d function
def convolution2d(img, kernel) -> np.ndarray:
    assert img.dtype == float, "Image must be in float"
    assert kernel.shape[0] == kernel.shape[1], "Kernel must be a square"
    assert kernel.shape[0] % 2 == 1, "Kernel size must be uneven"
    
    # calculate size of padding
    num_pad_pixels = int(kernel.shape[0]//2)
   
    
    img_padded = copy.deepcopy(img)
    img_padded = np.pad(img_padded, pad_width=([num_pad_pixels, ], [num_pad_pixels, ]))
    
    img_result = np.zeros_like(img)
    
    plt.imshow(img_padded, cmap='gray')
    plt.show()
    
    for u in range(num_pad_pixels, img_padded.shape[0] - num_pad_pixels):
        for v in range(num_pad_pixels, img_padded.shape[1] - num_pad_pixels):
            
            # vertical_patch = img_padded[u - num_pad_pixels: u + num_pad_pixels + 1, v - num_pad_pixels:v+num_pad_pixels+1]
            # vertical_patch = vertical_patch.flatten() * sobel_kernel_vertical.flatten()
            # vertical_sum = vertical_patch.sum()
            
            # horizontal_patch = img_padded[u - num_pad_pixels: u + num_pad_pixels + 1, v - num_pad_pixels : v + num_pad_pixels + 1]
            # horizontal_patch = horizontal_patch.flatten() * sobel_kernel_horizontal.flatten()
            # horizontal_sum = horizontal_patch.sum()
            
            patch = img_padded[u - num_pad_pixels: u + num_pad_pixels + 1, v - num_pad_pixels : v + num_pad_pixels + 1]
            patch = patch.flatten() * kernel.flatten()
            convolved = patch.sum()
            
            img_result[u - num_pad_pixels][v - num_pad_pixels] = convolved**2
            
    return img_result


def calculate_gradients(*, gray_img: np.ndarray, kernel_u: np.ndarray, kernel_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d_u = convolution2d(gray_img, sobel_kernel_horizontal)
    d_v = convolution2d(gray_img, sobel_kernel_vertical)
    d_mag = np.sqrt(d_u + d_v)
    d_angle = np.arctan(d_u / d_v)
    
    return d_u, d_v, d_mag, d_angle

d_u, d_v, d_mag, d_angle = calculate_gradients(gray_img=gray_img, kernel_u=sobel_kernel_horizontal, kernel_v=sobel_kernel_vertical)
_, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
figure_size = plt.gcf().get_size_inches()
plt.gcf().set_size_inches(3 * figure_size)
ax1.imshow(d_u, cmap="gray")
ax2.imshow(d_v, cmap="gray")
ax3.imshow(d_mag, cmap="gray")
ax4.imshow(d_angle, cmap="hsv")
plt.show()

result_own = convolution2d(img=gray_img, kernel=sobel_kernel_horizontal)
plt.imshow(result_own, cmap='gray')
plt.show()

result_scipy = convolve2d(gray_img, sobel_kernel_vertical, mode="same", fillvalue=0)
plt.imshow(result_scipy, cmap='gray')
plt.show()