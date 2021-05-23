''' In this file, you will utilize two parameters degree and include_bias.
    Reference https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PolynomialFeatures():
    
    def __init__(self, degree=2,include_bias=True):
        """
        Inputs:
        param degree : (int) max degree of polynomial features
        param include_bias : (boolean) specifies wheter to include bias term in returned feature array.
        """
        
        self.degree = degree
        self.include_bias = include_bias

    
    def transform(self,X):
        """
        Transform data to polynomial features
        Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. 
        For example, if an input sample is  np.array([a, b]), the degree-2 polynomial features with "include_bias=True" are [1, a, b, a^2, ab, b^2].
        
        Inputs:
        param X : (np.array) Dataset to be transformed
        
        Outputs:
        returns (np.array) Tranformed dataset.
        """
        # conver X to a list
        X = X.tolist()
        result = []

        # iterate over the length of X
        for b in range(len(X)):

            # change dataset accoring to bias
            if self.include_bias:
                X[b].insert(0, 1)
            
            # initialize an array to store dynamically all array of indices
            init_arr = []
            for j in range(len(X[b])):
                init_arr.append([j])

            # array of indices
            arr = [j for j in range(len(X[b]))]
            separate_arr = init_arr.copy()

            # iterate for the degree given
            for k in range(0,self.degree-1):
                # for len of the array containing indices
                for i in range(len(arr)):
                    temp = i
                    # this loop will have different length since length increases
                    for j in range((k)*len(arr),len(separate_arr)):
                        element = init_arr[j].copy()
                        element.append(temp)
                        init_arr.append(element) 
                separate_arr = init_arr.copy()
            # sort the array obtained to remove repeated elements
            array = []
            for m in range(len(init_arr)):
                init_arr[m].sort()
                if(init_arr[m] not in array):
                    array.append(init_arr[m])

            # calculate the final values by multiplying the numbers or columns at the place of indices
            final = []
            for i in array:
                lst = []
                # only if lenth satisfies the given degree
                if len(i)==self.degree:
                    for j in i:    
                        lst.append(X[b][j])        
                    final.append(np.product(lst))
            result.append(final)
        return result
        
        
        
        
        
        
    
                
                
