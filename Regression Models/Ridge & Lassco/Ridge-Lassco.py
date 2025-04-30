import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

car = pd.read_csv(r"C:\Users\Mahesh Babu\Downloads\car-mpg.csv")

car = car.drop(["car_name"], axis=1)
car["origin"] = car["origin"].replace({1: "america", 2: "europe", 3: "asia"})
car = pd.get_dummies(car, columns = ["origin"], dtype=int)
car = car.replace("?", np.nan)

car = car.apply(lambda x: x.fillna(x.median()), axis = 1)

x = car.drop(["mpg"], axis =1)
y = car[["mpg"]]

x_s = preprocessing.scale(x)
x_s = pd.DataFrame(x_s, columns=x.columns)

y_s = preprocessing.scale(y)
y_s = pd.DataFrame(y_s, columns=y.columns) 

x_train, x_test, y_train,y_test = train_test_split(x_s, y_s, test_size = 0.30, random_state = 1)
x_train.shape

regression_model = LinearRegression()
regression_model.fit(x_train, y_train)
for idx, col_name in enumerate(x_train.columns):
    print('The coefficient for {} is {}'.format(col_name, regression_model.coef_[0][idx]))
intercept = regression_model.intercept_[0]
print('The intercept is {}'.format(intercept))

ridge_model = Ridge(alpha = 0.3)
ridge_model.fit(x_train, y_train)
print('Ridge model coef: {}'.format(ridge_model.coef_))
#As the data has 10 columns hence 10 coefficients appear here    

lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(x_train, y_train)
print('Lasso model coef: {}'.format(lasso_model.coef_))

print(regression_model.score(x_train, y_train))
print(regression_model.score(x_test, y_test))

print('*************************')
#Ridge
print(ridge_model.score(x_train, y_train))
print(ridge_model.score(x_test, y_test))

print('*************************')
#Lasso
print(lasso_model.score(x_train, y_train))
print(lasso_model.score(x_test, y_test))

data_train_test = pd.concat([x_train, y_train], axis =1)
data_train_test.head()

import statsmodels.formula.api as smf
ols1 = smf.ols(formula = 'mpg ~ cyl+disp+hp+wt+acc+yr+car_type+origin_america+origin_europe+origin_asia', data = data_train_test).fit()
ols1.params

print(ols1.summary())

mse  = np.mean((regression_model.predict(x_test)-y_test)**2)
import math
rmse = math.sqrt(mse)
print('Root Mean Squared Error: {}'.format(rmse))

fig = plt.figure(figsize=(10,8))
sns.residplot(x= x_test['hp'], y= y_test['mpg'], color='green', lowess=True )
plt.show()

fig = plt.figure(figsize=(10,8))
sns.residplot(x= x_test['acc'], y= y_test['mpg'], color='green', lowess=True )
plt.show()

y_pred = regression_model.predict(x_test)

plt.scatter(y_test['mpg'], y_pred)
plt.show()