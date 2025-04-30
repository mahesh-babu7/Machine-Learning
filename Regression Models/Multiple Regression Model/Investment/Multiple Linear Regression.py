import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv(r"D:\VS Code\Machine Learning\Regression Models\Multiple Linear Regression\Investment\Investment.csv")

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

x = pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred  =regressor.predict(x_test)

m_coef = regressor.coef_
print(m_coef)

c_inter = regressor.intercept_
print(c_inter)

x = np.append(arr = np.ones((50,1)).astype(int), values=x, axis=1)

import statsmodels.api as sm
x_opt = x[:, [0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

import statsmodels.api as sm
x_opt = x[:, [0,1,2,3,5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

import statsmodels.api as sm
x_opt = x[:, [0,1,2,3]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

import statsmodels.api as sm
x_opt = x[:, [0,1,3]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

import statsmodels.api as sm
x_opt = x[:, [0,1]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()