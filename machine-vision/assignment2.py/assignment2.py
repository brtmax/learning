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
