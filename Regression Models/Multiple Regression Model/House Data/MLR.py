import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv(r'D:\VS Code\Machine Learning\Regression Models\Multiple Linear Regression\House Data\house_data.csv')

dataset.isnull().any()

dataset.dtypes

dataset = dataset.drop(['id','date'], axis = 1)

with sns.plotting_context("notebook",font_scale=2.5):
    g = sns.pairplot(dataset[['sqft_lot','sqft_above','price','sqft_living','bedrooms']],hue='bedrooms', palette='tab20',height=6)
g.set(xticklabels=[])
plt.show()

X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

import statsmodels.api as sm
x_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

import statsmodels.api as sm
x_opt = X[:, [0, 1, 2, 3,5,6,7,8,9,10,11,12,13,14,15,16,17]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()
