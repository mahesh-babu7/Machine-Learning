import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

data=pd.read_csv(r"D:\VS Code\Machine Learning\Regression Models\Polynomial Regression\emp_sal.csv")

x=data.iloc[:,1:2].values # INDEPENDENT VARIABL
y=data.iloc[:,2].values # DEPENDENT VARIABLE

# KKN REGRESSION MODEL
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model = KNeighborsRegressor(n_neighbors=5, weights='distance', p=2)
knn_reg_model.fit(x,y)

# KNN Regression Visualization
plt.scatter(x,y,color='red')
plt.plot(x,knn_reg_model.predict(x),color='blue')
plt.title('KNN Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# KNN Regression Predictions
knn_reg_pred = knn_reg_model.predict([[6.5]])
print(knn_reg_pred)
