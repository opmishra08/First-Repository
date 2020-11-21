#Loading required modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#Loading Dataset
iris= datasets.load_iris()

#printing description feature and labels
# print(iris.DESCR)
features= iris.data
labels = iris.target
# print(features[0],labels[0])

#Training the classifires
clf= KNeighborsClassifier()
clf.fit(features,labels)

preds = clf.predict([[3.1,4.5,6.4,788]])
print(preds)