import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

print(df.head())

prod_per_year = df.groupby('year').totalprod.mean().reset_index()

X = df['year']
X = X.values.reshape(-1, 1)

y = df['totalprod']

plt.scatter(X, y)
plt.show

regr = linear_model.LinearRegression()

regr.fit(X,y)

print(regr.coef_[0])

y_predict = regr.predict(X)

plt.plot(X, y_predict)
plt.show

X_future = np.array(range(1, 11))
X_future = X_future.reshape(-1, 1)

future_predict = regr.predict(X_future)

plt.plot(future_predict, X_future)
plt.show