# test classification dataset
from sklearn.datasets import make_classification

# define dataset
x, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)

print(x.shape, y.shape)