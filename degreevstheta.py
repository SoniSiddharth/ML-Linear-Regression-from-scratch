import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures

np.random.seed(42)

def normal_regression(X,y):
    X_transpose = np.transpose(X)
    A = np.linalg.inv(X_transpose.dot(X))
    B = X_transpose.dot(y)
    return A.dot(B)

lst = []
degrees = [1,3,5,7,9]
sample_size = []
l = 0
for N in range (10,200,40):
    # x = np.random.rand(N)
    x = np.array([i*np.pi/180 for i in range(N,300,4)])
    y = 4*x + 7 + np.random.normal(0,3,len(x))
    x = np.array(np.matrix(x).transpose())
    temp = []
    for degree in degrees:
        poly = PolynomialFeatures(degree,include_bias=True)
        X = poly.transform(x)
        coeff = normal_regression(X,y)
        temp.append(np.log(np.linalg.norm(np.array(coeff))))
    lst.append(temp)
    l+=1
    sample_size.append(len(x))

for i in range (1,l+1):
    plt.plot(degrees,lst[i-1],label='Num of Samples '+str(sample_size[i-1]))
    plt.xlabel("Value of degree")
    plt.ylabel("Log of L2 norm of coefficients")
plt.legend(loc = 'best')
plt.savefig('./images/q6plot.png')
plt.show()