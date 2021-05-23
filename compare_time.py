import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time

# np.random.seed(42)

grad = []
normal = []
num_features = []

N = 30
for i in range (50,1000, 5):
	X = pd.DataFrame(np.random.randn(N, i))
	y = pd.Series(np.random.randn(N))

	LR = LinearRegression(fit_intercept=True)	
	start = time.time()
	LR.fit_vectorised(X, y)
	grad.append(time.time()-start)

	LR_normal = LinearRegression(fit_intercept=True)
	start_time = time.time()
	LR_normal.fit_normal(X, y) 
	normal.append(time.time()-start_time)
	num_features.append(i)

plt.plot(num_features, grad, label = 'Gradient Descent')
plt.plot(num_features, normal, label = 'Normal Equation')
plt.xlabel('Num of features')
plt.ylabel('time in seconds')
plt.legend(loc = 'best')
plt.savefig('./images/q8features.png')
plt.show()

grad = []
normal = []
num_samples = []
P = 20

for i in range (50,2000, 5):
	X = pd.DataFrame(np.random.randn(i, P))
	y = pd.Series(np.random.randn(i))

	LR = LinearRegression(fit_intercept=True)	
	start = time.time()
	LR.fit_vectorised(X, y)
	grad.append(time.time()-start)

	LR_normal = LinearRegression(fit_intercept=True)
	start_time = time.time()
	LR_normal.fit_normal(X, y) 
	normal.append(time.time()-start_time)
	num_samples.append(i)

plt.plot(num_samples, grad, label = 'Gradient Descent')
plt.plot(num_samples, normal, label = 'Normal Equation')
plt.xlabel('Num of samples')
plt.ylabel('time in seconds')
plt.legend(loc = 'best')
plt.savefig('./images/q8samples.png')
plt.show()