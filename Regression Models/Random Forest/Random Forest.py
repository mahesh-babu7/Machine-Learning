import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

data=pd.read_csv(r"D:\VS Code\Machine Learning\Regression Models\Random Forest Model\emp_sal.csv")

x=data.iloc[:,1:2].values # INDEPENDENT VARIABL
y=data.iloc[:,2].values # DEPENDENT VARIABLE

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators=30,random_state=0)
regressor_rf.fit(x,y)

# Random Forest Visualization
plt.scatter(x,y,color='red')
plt.plot(x,regressor_rf.predict(x),color='blue')
plt.title('Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Random Forest Predictions
y_pred_df = regressor_rf.predict([[6.5]])
print(y_pred_df)