import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)

N = 30

print("----------------------------------- Multi collinear ----------------------------------")

P = 4
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

X[P] = X.iloc[:][P-1]*6
# print(X)

LR = LinearRegression(fit_intercept=True)
LR.fit_vectorised(X, y)
y_hat = LR.predict(X)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))

print("----------------------------------------- Normal dataset -------------------------------------")

P = 5
Xnew = pd.DataFrame(np.random.randn(N, P))
ynew = pd.Series(np.random.randn(N))
# print(Xnew)

LRnew = LinearRegression(fit_intercept=True)
LRnew.fit_vectorised(Xnew, ynew)
y_hatnew = LRnew.predict(Xnew)
print('RMSE: ', rmse(y_hatnew, ynew))
print('MAE: ', mae(y_hatnew, ynew))