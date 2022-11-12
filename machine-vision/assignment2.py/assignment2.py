import numpy as np
import skimage.io
import os
import copy
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


gray_img = plt.imread("/Users/maxbretschneider/Desktop/Development/learning/machine-vision/assignment2.py/sources/postit2g.png")


print(gray_img.shape)

# ensure image is float type
gray_img = gray_img.astype(float)


# implement convolution2d function
def convolution2d(*, img, kernel) -> np.ndarray:
    assert img.dtype == float, "Image must be in float"
    # assert kernel.dtype == float, "Kernel must be in float"
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
            
            vertical_patch = img_padded[u - num_pad_pixels: u + num_pad_pixels + 1, v - num_pad_pixels:v+num_pad_pixels+1]
            vertical_patch = vertical_patch.flatten() * sobel_kernel_vertical.flatten()
            vertical_sum = vertical_patch.sum()
            
            horizontal_patch = img_padded[u - num_pad_pixels: u + num_pad_pixels + 1, v - num_pad_pixels : v + num_pad_pixels + 1]
            horizontal_patch = horizontal_patch.flatten() * sobel_kernel_horizontal.flatten()
            horizontal_sum = horizontal_patch.sum()
            
            img_result[u - num_pad_pixels][v - num_pad_pixels] = np.sqrt(vertical_sum**2 + horizontal_sum**2)
            
    return img_result

# define the sobel kernel
sobel_kernel_vertical = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
sobel_kernel_horizontal = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])

result_own = convolution2d(img=gray_img, kernel=sobel_kernel_horizontal)

plt.imshow(result_own, cmap='gray')
plt.show()



result_scipy = convolve2d(gray_img, sobel_kernel, mode="same", fillvalue=0)
assert np.allclose(result_own, result_scipy), "Incorrect implementation of convolution2d"