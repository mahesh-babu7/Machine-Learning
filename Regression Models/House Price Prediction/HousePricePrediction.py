import numpy as np
import pandas as pd

df = pd.read_csv(r'D:\VS Code\Regression Models\House Price Predicition\House_data.csv')

x = np.array(df['sqft_living']).reshape(-1, 1)
y = np.array(df['price'])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

Predicitions = model.predict(x_test)

import matplotlib.pyplot as plt

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, model.predict(x_train), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

plt.scatter(x_test, y_test, color= 'red')
plt.plot(x_train, model.predict(x_train), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

import pickle
filename = 'HousePricePredictionModel.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
    
