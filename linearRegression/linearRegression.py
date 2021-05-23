import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
import autograd.numpy as np
from autograd import elementwise_grad
import autograd as grad

def cost_function(theta,X,y):
    error = ((y - np.dot(X,theta))**2)/len(X)
    return error

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.data_used = None
        self.thetas = []
        self.theta_history = []

    def fit_non_vectorised(self, X, y, batch_size=5, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        dataX = np.array(X)
        dataY = np.array(y)

        # add biases if fit_intercept is true
        if self.fit_intercept:
            bias = np.ones((dataX.shape[0], 1))
            dataX = np.append(bias, dataX, axis=1)
        dataY = dataY.reshape(len(y), 1)

        # creating one single dataset including y
        self.data_used = dataX
        data = np.hstack((dataX, dataY))
        # print(data)

        # defining coefficients (theta)
        self.coef_ = [0 for j in range(dataX.shape[1])]

        # running for each iteration
        for itr in range(1, n_iter+1):
            error = []
            # iteration for batches
            for start_index in range(0, data.shape[0], batch_size):
                # print(" start index ", start_index)
                for row in range(start_index, min(start_index+batch_size, data.shape[0])):
                    prediction = 0
                    # predicting y values
                    for col in range(data.shape[1]-1):
                        prediction += self.coef_[col]*data[row][col]
                    error.append(data[row][-1]-prediction)
                
                # updating coefficients
                for col in range(0, data.shape[1]-1):
                    mse = 0
                    N = 0
                    for row in range(start_index, min(start_index+batch_size, data.shape[0])):
                        mse += error[row]*(data[row][col])*(-1)
                        N+=1
                    # updating based on lr type
                    if (lr_type=="constant"):
                        self.coef_[col] -= 2*lr*(mse/N)
                    else:
                        self.coef_[col] -= 2*(lr/itr)*(mse/N)

    def fit_vectorised(self, X, y,batch_size=5, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        dataX = np.array(X)
        dataY = np.array(y)

        # adding bias if given
        Nsamples = dataX.shape[0]
        if self.fit_intercept:
            bias = np.ones((Nsamples, 1))
            dataX = np.append(bias, dataX, axis=1)

        dataY = np.array(y.copy())
        dataY = dataY.reshape(len(y),1)
        # print(dataY)

        self.data_used = dataX.copy()

        # initializing theta
        self.coef_ = np.array([[0.0] for j in range(dataX.shape[1])])
        # print(self.coef_)
        batches_epoch = Nsamples//batch_size

        # dividing iterations so that coefficients are updated number of iteration times
        for itr in range(n_iter//batches_epoch):
            for bch in range(0, dataX.shape[0], batch_size):
                # predict y values
                y_pred = dataX[bch:min(Nsamples, bch+batch_size),:].dot(self.coef_)
                # error calculation
                error = y_pred - dataY[bch:min(Nsamples, bch+batch_size),:]
                objective = (dataX[bch:min(Nsamples, bch+batch_size),:]).T.dot(error)
                # updating coefficients
                if (lr_type=="constant"):
                    self.coef_ = self.coef_ - (2/len(y_pred))*(lr)*objective
                else:
                    self.coef_ = self.coef_ - (2/len(y_pred))*(lr/batch_size)*objective

                # storing theta history for plots
                self.theta_history.append(self.coef_)

    def fit_autograd(self, X, y, batch_size=5, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
 
        dataX = np.array(X)
        dataY = np.array(y)

        # adding bias if given 
        Nsamples = dataX.shape[0]
        if self.fit_intercept:
            bias = np.ones((Nsamples, 1))
            dataX = np.append(bias, dataX, axis=1)

        dataY = np.array(y.copy())
        dataY = dataY.reshape(len(y),1)
        # print(dataY)

        # coefficients initialization
        self.data_used = dataX.copy()
        self.coef_ = np.array([[0.0] for j in range(dataX.shape[1])])
        batches_epoch = Nsamples//batch_size

        # each iteration
        for iter in range(n_iter//batches_epoch):
            for bch in range(0, dataX.shape[0], batch_size):
                # inbuilt autograd function
                agrad = elementwise_grad(cost_function)     
                X_set = dataX[bch:min(Nsamples, bch+batch_size),:]
                Y_set = dataY[bch:min(Nsamples, bch+batch_size),:]
                objective = agrad(self.coef_, X_set, Y_set)
                if (lr_type=="constant"):
                    self.coef_ = self.coef_ - 2*(lr)*objective
                else:
                    self.coef_ = self.coef_ - 2*(lr/batch_size)*objective

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''
        # take transpose
        X_transpose = np.transpose(X)
        # take inverse after dot product
        A = np.linalg.inv(X_transpose.dot(X))
        B = X_transpose.dot(y)
        # get final coefficients
        self.coef_ = A.dot(B)

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        y_predict = self.data_used.dot(self.coef_)
        y_predict = y_predict.tolist()
        return (pd.Series(y_predict))
    
    # helper function for plots
    def predict_plot(self, d_set, theta):
        return list(d_set.dot(theta))

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """
        # this plot is in q7_plot_contour.py
        pass

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        # this plot is in q7_plot_contour.py
        pass

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """

        dataX = np.array(X)
        dataY = np.array(y)

        Nsamples = dataX.shape[0]
        if self.fit_intercept:
            bias = np.ones((Nsamples, 1))
            dataX = np.append(bias, dataX, axis=1)

        dataY = np.array(y.copy())
        dataY = dataY.reshape(len(y),1)

        self.data_used = dataX.copy()
        self.coef_ = np.array([[0.0] for j in range(dataX.shape[1])])

        batch_size = 40
        n_iter = 100
        lr = 0.008
        lr_type = "constant"
        batches_epoch = Nsamples//batch_size

        for epoch in range(n_iter//batches_epoch):
            for bch in range(0, dataX.shape[0], batch_size):
                y_pred = dataX[bch:min(Nsamples, bch+batch_size),:].dot(self.coef_)
                error = y_pred - dataY[bch:min(Nsamples, bch+batch_size),:]
                objective = (dataX[bch:min(Nsamples, bch+batch_size),:]).T.dot(error)
                if (lr_type=="constant"):
                    self.coef_ = self.coef_ - (2/len(y_pred))*(lr)*objective
                else:
                    self.coef_ = self.coef_ - (2/len(y_pred))*(lr/epoch)*objective
                self.thetas.append(self.coef_)
        return 
