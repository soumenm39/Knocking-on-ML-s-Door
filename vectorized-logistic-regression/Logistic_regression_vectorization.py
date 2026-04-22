# This Python 3 environment comes with many helpful analytics libraries installed
#basic logistic regression

#import necessary library
import pandas as pd  #for reading CSV file
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import matplotlib.pyplot as plt

data = pd.read_csv('Exam_Data.csv')
data['GIExam'] = np.array(data.GIExam)
data['SubExam'] = np.array(data.SubExam)
data['Output'] = np.array(data.Output)

#pre processing of data
input = np.column_stack((data['GIExam'], data['SubExam']))
lebel = np.array(data['Output']).reshape(-1,1)
ones = np.ones((input.shape[0], 1))
input_new = np.hstack((ones, input))
X = np.array(input_new)
# store mean and std for later (important for boundary)
mean = np.mean(X[:,1:], axis=0)
std  = np.std(X[:,1:], axis=0)
# Normalize features (except bias)
X[:,1:] = (X[:,1:] - mean) / std
#X[:,1:] = (X[:,1:] - np.mean(X[:,1:], axis=0)) / np.std(X[:,1:], axis=0)
#W = np.random.randn(3,1)*0.006
#W = np.ones((3,1))*0.01

W = np.array([-0.02, -0.16, 0.2])
W = W.reshape(-1,1)


#defing sigmoid function
def sigmoid_func(W,X):
  Z = X @ W
  Z = np.clip(Z, -500, 500)
  S = 1/(1 + np.exp(-Z))
  return S

#Define loss function
def error(W,X,lebel):
  epsilon = 1e-10
  sigmoid = sigmoid_func(W,X)
  sigmoid = np.clip(sigmoid, epsilon, 1-epsilon)
  error_1 = np.log(sigmoid).T @ lebel  #first terms of loss function
  error_2 = np.log(1 - sigmoid).T @ (1 - lebel)
  error = -(1/(lebel.shape[0]))*(error_1 + error_2)
  return error

#defining gradient descent
def gradient_descent(W,X,lebel,L):
  diff = sigmoid_func(W,X) - lebel
  #print(diff.shape)
  M = X.T @ diff
  n = lebel.shape[0]
  W = W - (L*M)/n
  return W

#convert W to original 
def convert_weights(W, mean, std):
    c, m1, m2 = W.flatten()
    m1_new = m1 / std[0]
    m2_new = m2 / std[1]
    c_new = c - (m1*mean[0]/std[0]) - (m2*mean[1]/std[1])
    W_original = np.array([[c_new],[m1_new],[m2_new]])
    return W_original

loss = []
epo  = []
epochs = 1000
L = 0.0075

plt.pause(2)

#learning part
for i in range(epochs):
    if  i < 200:
        # convert normalized weights to original
        W_original = convert_weights(W, mean, std)
        c, m1, m2 = W_original.flatten()

        # decision boundary
        x_vals = np.linspace(data['GIExam'].min(), data['GIExam'].max(), 100)
        y_vals = -(c + m1*x_vals)/m2

        # scatter plot
        fail = data[data['Output']==0]
        pass_ = data[data['Output']==1]

        plt.scatter(fail['GIExam'], fail['SubExam'], color='red', label='Fail')
        plt.scatter(pass_['GIExam'], pass_['SubExam'], color='green', label='Pass')

        # plot boundary
        plt.plot(x_vals, y_vals, color='blue')

        plt.title(f"Iteration {i}")
        plt.xlabel('GIExam')
        plt.ylabel('SubExam')
        plt.legend()
        #plt.pause(0.5)
        plt.draw()
        plt.pause(0.3)
        plt.cla()       
    loss.append(error(W,X,lebel).item())
    epo.append(i)
    W = gradient_descent(W,X,lebel,L)

#print the final value of m and c
print(W)


W_original = convert_weights(W, mean, std)
c, m1, m2 = W_original.flatten()

#Final decision boundary
x_vals = np.linspace(data['GIExam'].min(), data['GIExam'].max(), 100)
y_vals = -(c + m1*x_vals)/m2

# scatter plot
fail = data[data['Output']==0]
pass_ = data[data['Output']==1]
plt.scatter(fail['GIExam'], fail['SubExam'], color='red', label='Fail')
plt.scatter(pass_['GIExam'], pass_['SubExam'], color='green', label='Pass')

# plot final boundary boundary
plt.plot(x_vals, y_vals, color='blue')

plt.title(f"Iteration {i}")
plt.xlabel('GIExam')
plt.ylabel('SubExam')
plt.legend()
plt.show(block=False)   # important
plt.pause(3.3)
plt.close()


#uncommeting the following section will show the plot of Loss vs epochs 
plt.figure(figsize=(5,5))
plt.plot(epo, loss, color="red")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss vs Epoch")
plt.show()

