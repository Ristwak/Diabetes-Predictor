import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()     #line loads the diabetes dataset from scikit-learn
# print(diabetes.keys())      #Keys will tell us what dictionary keys are in there
# We are doing this to get the data and plot a graph

#We are slicing the diabetes variable
# This line extracts a specific feature from the diabetes dataset. It selects the third feature (index 2) and reshapes it using np.newaxis to have a new axis. This is done because scikit-learn expects data to be in a specific format
# diabetes_X = diabetes.data[:, np.newaxis, 2]
# diabetes_X = diabetes.data


# These valuesa are example to check the function and this example is written in the copy as well
diabetes_X = np.array([[1],[2],[3]])
diabetes_X_train = diabetes_X    #This line means we have to take 30 values from the last
diabetes_X_test = diabetes_X   #This line means we have to take 20values from the start where : sign means all

diabetes_Y_train = np.array([3,2,4])   # This is the label for the value of diabetes_X_train to plot the graph
diabetes_Y_test = np.array([3,2,4])

model = linear_model.LinearRegression()

model.fit(diabetes_X_train,diabetes_Y_train)

diabetes_Y_predicted = model.predict(diabetes_X_test)

# This line calculates the mean squared error (MSE) between the actual target values (diabetes_Y_test) and the predicted values (diabetes_Y_predicted).
# The MSE is a measure of how well the model's predictions match the actual values.
# print("The mean swuared value is : ",mean_squared_error(ACTUAL VALUE, PREDICTED VALUE))
print("The mean swuared value is : ",mean_squared_error(diabetes_Y_test, diabetes_Y_predicted))
print("Weights : ",model.coef_)    #Weight is tan theta which is the slope
print("Intercept : ",model.intercept_)

# This line creates a scatter plot of the testing data points. It shows the actual data points as individual dots on the plot.
plt.scatter(diabetes_X_test,diabetes_Y_test)

# This line creates a line plot of the predicted values (diabetes_Y_predicted) based on the testing data (diabetes_X_test).
plt.plot(diabetes_X_test,diabetes_Y_predicted)

# This line displays the plot on the screen
plt.show()

# The mean swuared value is :  2561.320427728385
# Weights :  [941.43097333]
# Intercept :  153.39713623331644

