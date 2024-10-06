from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names