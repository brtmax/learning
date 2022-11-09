from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# Gaussian Process Classifier
model = GaussianProcessClassifier(kernel=1*RBF(1.0))