# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 18:41:05 2024

@author: Akshatha
"""

from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target
print("Shape of X:")
print(X.shape)
print("Shape of y:")
print(y.shape)


#Slicing of the data set
x1 = X[10:20, 1:3]
print(x1.shape)
x2 = X[:40, 1:]
print(x2.shape)
x3 = X[110:, :]
print(x3.shape) 


#mean,median and standard deviation of each of the feature
feature1 = X[:,0]
np.mean(feature1, axis=0)
np.median(feature1,axis=0)
np.std(feature1,axis=0)

feature2 = X[:,1]
np.mean(feature2, axis=0)
np.median(feature2,axis=0)
np.std(feature2,axis=0)


feature3 = X[:,2]
np.mean(feature3, axis=0)
np.median(feature3,axis=0)
np.std(feature3,axis=0)

feature4 = X[:,3]
np.mean(feature4,axis=0)
np.mean(feature4, axis=0)
np.median(feature4,axis=0)
np.std(feature4,axis=0)


#histogram plotting
plt.hist(feature1)
plt.title('Histogram of Feature 1 in Iris Dataset')
plt.xlabel('Feature 1 Values')
plt.ylabel('Frequency')

plt.hist(feature2)
plt.title('Histogram of Feature 2 in Iris Dataset')
plt.xlabel('Feature 2 Values')
plt.ylabel('Frequency')

plt.hist(feature3)
plt.title('Histogram of Feature 3 in Iris Dataset')
plt.xlabel('Feature 3 Values')
plt.ylabel('Frequency')

plt.hist(feature4)
plt.title('Histogram of Feature 4 in Iris Dataset')
plt.xlabel('Feature 4 Values')
plt.ylabel('Frequency')


#Scatter plot of feature 1 and 3
plt.scatter(feature1, feature3, c=y, cmap='viridis', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 3')
plt.title('Scatter Plot of Features 1 and 3 with Class Colors')

#Scatter plot of feature 2 and 4
plt.scatter(feature2, feature4, c=y, cmap='viridis', edgecolors='k')
plt.xlabel('Feature 2')
plt.ylabel('Feature 3')
plt.title('Scatter Plot of Features 2 and 4 with Class Colors')