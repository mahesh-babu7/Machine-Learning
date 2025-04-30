import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"D:\VS Code\Machine Learning\Regression Models\Simple Linear Regression\Salary Prediction\Salary_Data.csv")

x = dataset.iloc[:, :-1]  
y = dataset.iloc[:, -1] 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

# plot the graph

plt.scatter(x_test, y_test, color = 'red')  # Real salary data (testing)
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  # Regression line from training set
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# i want to predict the future 

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

pred_12yr_emp_exp = m_slope * 12 + c_intercept
print(pred_12yr_emp_exp)

pred_20yr_emp_exp = m_slope * 20 + c_intercept
print(pred_20yr_emp_exp)

# Bais and Variance

bias = regressor.score(x_train, y_train)
print(bias)


variance = regressor.score(x_test, y_test)
print(variance)

# Stats for ML

dataset.mean()

dataset['Salary'].mean()

dataset.median()

dataset["Salary"].median()

dataset.mode()

dataset["Salary"].mode()

dataset.var()

dataset["Salary"].var()

dataset.std()

dataset["Salary"].std()

dataset.corr()


# SSR
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

# SSE
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

#SST
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

#r2
r_square = 1 - SSR/SST
print(r_square)

import pickle

filename = 'linear_regression_model.pkl'

with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
    
print("Model has been pickled and saved as linear_regression_model.pkl")




