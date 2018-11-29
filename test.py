# __Author__:Zcc
from chi_merge import ChiMerge
from sklearn.datasets import load_iris

iris = load_iris()

cm = ChiMerge(num_features=[0, 1, 2, 3], cat_features=[])
cm.fit(iris.data, iris.target)
cm.transform(iris.data)
print(cm.split_point)










