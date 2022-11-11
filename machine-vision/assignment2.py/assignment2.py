import numpy as np
import skimage.io
import os
import copy
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


gray_img = plt.imread("/Users/maxbretschneider/Desktop/Development/learning/machine-vision/assignment2.py/sources/postit2g.png")
plt.imshow(gray_img, cmap='gray')
plt.show()

print(gray_img.shape)

# ensure image is float type
gray_img = gray_img.astype(float)


# implement convolution2d function
def convolution2d(*, img, kernel) -> np.ndarray:
    assert img.dtype == float, "Image must be in float"
    assert kernel.dtype == float, "Kernel must be in float"
    assert kernel.shape[0] == kernel.shape[1], "Kernel must be a square"
    assert kernel.shape[0] % 1 == 1, "Kernel size must be uneven"
    
    # calculate size of padding
    num_pad_pixels = (kernel.shape[0] - 1) / 2
    
    img_padded = copy.deepcopy(img)
    img_padded = np.pad(img_padded, num_pad_pixels)
    
    img_result = np.zeros_like(img)
    
    # range over v dimension
    for v in range(0,img.shape[0]):
        # range over u dimension
        for u in range(0, img.shape[1]):
            patch = img_padded[u:kernel[0], v:kernel[1]]
            
            img_result[u:kernel[0], v:kernel[1]] = patch
    return img_result