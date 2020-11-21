from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

mnist= fetch_openml('mnist_784')
x,y = mnist['data'],mnist['target']
print(x.shape)
print(y.shape)
# %matplotlib inline
some_digit = x[3600]
some_digit_image = some_digit.reshape(28,28) # lets reshape it to plot it
plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation='nearest')
plt.axis('off')
print(y[3601])
x_train, x_test = x[0:6000], x[6000:7000]
y_train, y_test = y[0:6000], y[60000:7000]
shuffle_index=np.random.permutation(6000)
x_train,y_train = x_train[shuffle_index],y_train[shuffle_index]
y_train=y_train.astype(np.int8)
y_test=y_test.astype(np.int8)
y_train_2=(y_train!=2)
y_test_2=(y_test!=2)
print(y_test_2)
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(tol=0.1)
clf.fit(x_train,y_train_2)
clf.predict([some_digit])
from sklearn.model_selection import cross_val_score
a=cross_val_score(clf, x_train, y_train,cv=3,scoring="accuracy")
print(a.mean())
