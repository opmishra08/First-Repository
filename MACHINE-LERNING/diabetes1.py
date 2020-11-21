import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error 


diabetes = datasets.load_diabetes()
# print(diabetes.keys())
# diabetes_x = diabetes.data
# print(diabetes.DESCR)
# diabetes_x = diabetes.data[:,np.newaxis,2]
# print(diabetes_x)
# diabetes_x_train = diabetes_x[:-30]
# diabetes_x_test = diabetes_x[-30:]
# diabetes_y_train = diabetes.target[:-30]
# diabetes_y_test = diabetes.target[-30:]

diabetes_x =np.array([[1],[2],[3]])
diabetes_x_train = diabetes_x
diabetes_x_test = diabetes_x
# diabetes_y_train = diabetes.target
# diabetes_y_test = diabetes.target
diabetes_y_train = np.array([3,2,4])
diabetes_y_test =np.array([3,2,4])

model= linear_model.LinearRegression()
model.fit(diabetes_x_train,diabetes_y_train)
diabetes_y_predicted = model.predict(diabetes_x_test)
print("Mean Squarred error is: ", mean_squared_error(diabetes_y_test,diabetes_y_predicted))
print("Weights: " , model.coef_)
print("Intercept: " , model.intercept_)
plt.scatter(diabetes_x_test,diabetes_y_test)
plt.plot(diabetes_x_test,diabetes_y_predicted)
plt.show()