import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

data=pd.read_csv(r"D:\VS Code\Machine Learning\Regression Models\SVR Model\emp_sal.csv")

x=data.iloc[:,1:2].values # INDEPENDENT VARIABL
y=data.iloc[:,2].values # DEPENDENT VARIABLE

# SVR MODEL
from sklearn.svm import SVR
svr_reg=SVR(kernel="poly", degree=6)
svr_reg.fit(x,y)

# SVR Regression Visualization
plt.scatter(x,y,color='red')
plt.plot(x,svr_reg.predict(x),color='blue')
plt.title('SVR Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# SVR Regression Predictions
svr_pred=svr_reg.predict([[6.5]])
print(svr_pred)