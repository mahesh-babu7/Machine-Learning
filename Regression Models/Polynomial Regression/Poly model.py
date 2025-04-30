import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

data=pd.read_csv(r"D:\VS Code\Machine Learning\Regression Models\Polynomial Regression\emp_sal.csv")

x=data.iloc[:,1:2].values # INDEPENDENT VARIABL
y=data.iloc[:,2].values # DEPENDENT VARIABLE

# Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=5)
x_poly=poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

# Polynomial Regression Visualization
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color="blue")
plt.title('truth or BLuff (polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Polynomial Regression Prediction
poly_model_pred=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)

