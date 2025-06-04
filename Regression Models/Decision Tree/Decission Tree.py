import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

data=pd.read_csv(r"D:\VS Code\Machine Learning\Regression Models\Decission Tree Model\emp_sal.csv")

x=data.iloc[:,1:2].values # INDEPENDENT VARIABL
y=data.iloc[:,2].values # DEPENDENT VARIABLE

# Decission tree regression
from sklearn.tree import DecisionTreeRegressor
regressor_dtr = DecisionTreeRegressor(criterion="absolute_error", splitter="best")
regressor_dtr.fit(x,y)

# Decission tree visualization
plt.scatter(x,y,color='red')
plt.plot(x,regressor_dtr.predict(x),color='blue')
plt.title('Decission Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Decission tree Predictions
y_pred_dtr = regressor_dtr.predict([[6.5]])
print(y_pred_dtr)