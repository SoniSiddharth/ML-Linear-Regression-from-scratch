import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

print("------------------------------ Non-Vectorised Gradient Descent ------------------------------------")

for fit_intercept in [True, False]:
    print("Bias is ", fit_intercept)
    for l_type in ["constant", "inverse"]:
        print("Learning rate is ", l_type)
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit_non_vectorised(X, y, lr_type=l_type) # here you can use fit_non_vectorised / fit_autograd methods
        y_hat = LR.predict(X)
        print('RMSE: ', rmse(y_hat, y))
        print('MAE: ', mae(y_hat, y))
    print()

print("---------------------------------- Vectorised Gradient Descent -----------------------")

for fit_intercept in [True, False]:
    print("Bias is ", fit_intercept)
    for l_type in ["constant", "inverse"]:
        print("Learning rate is ", l_type)
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit_vectorised(X, y, lr_type=l_type) # here you can use fit_non_vectorised / fit_autograd methods
        y_hat = LR.predict(X)
        print('RMSE: ', rmse(y_hat, y))
        print('MAE: ', mae(y_hat, y))
    print()

print("-------------------------------------- Autograd Regression -----------------------------")

for fit_intercept in [True, False]:
    print("Bias is",fit_intercept)
    for l_type in ["constant", "inverse"]:
        print("Learning rate is ", l_type)
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit_autograd(X, y, lr_type=l_type) # here you can use fit_non_vectorised / fit_autograd methods
        y_hat = LR.predict(X)
        print('RMSE: ', rmse(y_hat, y))
        print('MAE: ', mae(y_hat, y))
    print()