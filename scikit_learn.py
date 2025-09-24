from sklearn.datasets import load_iris, fetch_california_housing

iris = load_iris()
print(iris.data.shape, iris.target.shape)

housing = fetch_california_housing()
print(housing.data.shape, housing.target.shape)

#to show how different sized datasets can be loaded