import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))
# print(x)

def normal_regression(X,y):
    X_transpose = np.transpose(X)
    A = np.linalg.inv(X_transpose.dot(X))
    B = X_transpose.dot(y)
    return A.dot(B)

arr_norm = []
degrees = [i+1 for i in range(9)]
x = np.array(np.matrix(x).transpose())

include_bias = True
for degree in degrees:
    poly = PolynomialFeatures(degree,include_bias = include_bias)
    X = poly.transform(x)
    coeff = normal_regression(X,y)
    arr_norm.append(np.linalg.norm(coeff))

plt.plot(degrees, arr_norm)
plt.xlabel("Degree of the polynomial")
plt.ylabel("Magnitude of coefficient (theta)")
plt.savefig('./images/q5plot.png')
plt.show()
