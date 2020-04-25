#!/usr/bin/env python
# coding: utf-8

# # Numpy, Matplotlib and Sklearn Tutorial


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# download and load mnist data from https://www.openml.org/d/554
# for this tutorial, the data have been downloaded already in './scikit_learn_data'
X, Y = fetch_openml('mnist_784', version=1, data_home='./scikit_learn_data', return_X_y=True)
# make the value of pixels from [0, 255] to [0, 1] for further process
X = X / 255.
# split data to train and test (for faster calculation, just use 1/10 data)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)


# #### Q1:
# Please use the logistic regression(default parameters) in sklearn to classify the data above, and print the training accuracy and test accuracy.

# In[ ]:


# TODO:use logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

cls=LogisticRegression()
cls.fit(X_train,Y_train)
train_accuracy=cls.score(X_train,Y_train)
test_accuracy=cls.score(X_test,Y_test)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))


# #### Q2:
# Please use the naive bayes(Bernoulli, default parameters) in sklearn to classify the data above, and print the training accuracy and test accuracy.

# In[ ]:


# TODO:use naive bayes
from sklearn.naive_bayes import BernoulliNB
nb=BernoulliNB()
nb.fit(X_train,Y_train)
train_accuracy=nb.score(X_train,Y_train)
test_accuracy=nb.score(X_test,Y_test)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))


# #### Q3:
# Please use the support vector machine(default parameters) in sklearn to classify the data above, and print the training accuracy and test accuracy.

# In[ ]:


# TODO:use support vector machine
from sklearn.svm import LinearSVC
LSVC=LinearSVC()
LSVC.fit(X_train,Y_train)
train_accuracy=LSVC.score(X_train,Y_train)
test_accuracy=LSVC.score(X_test,Y_test)


print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))


# #### Q4:
# Please adjust the parameters of SVM to increase the testing accuracy, and print the training accuracy and test accuracy.

# In[ ]:


# TODO:use SVM with another group of parameters
L2=LinearSVC(C=0.01)
L2.fit(X_train,Y_train)
train_accuracy=L2.score(X_train,Y_train)
test_accuracy=L2.score(X_test,Y_test)


print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))
