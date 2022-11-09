from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel


from sklearn.datasets import load_iris
from sklearn.gaussian_process.kernels import RBF

# define dataset
x, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)

# Define model
# Gaussian Process Classifier
model = GaussianProcessClassifier()

# Define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Define grid
grid = dict()

grid['kernel'] = [1*RBF(), 1*DotProduct(), 1*Matern(), 1*RationalQuadratic(), 1*WhiteKernel()]
