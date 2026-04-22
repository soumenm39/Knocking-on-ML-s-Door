# This Python 3 environment comes with many helpful analytics libraries installed
#basic linear regression

#import necessary library
import pandas as pd  #for reading CSV file
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import matplotlib.pyplot as plt

start = time.time()

loss = []
epo  = []
epochs = 2000
L = 0.004

data = pd.read_csv('2D_regression_Data.csv')
data['biking'] = np.array(data.biking)     #(data.biking - data['biking'].mean())/(data['biking'].max() - data['biking'].min())
data['smoking'] = np.array(data.smoking)   #(data.smoking - data['smoking'].mean())/(data['smoking'].max() - data['smoking'].min())
data['heart'] = np.array(data.heart)       #(data.heart - data['heart'].mean())/(data['heart'].max() - data['heart'].min())

#pre processing of data
input = np.column_stack((data['biking'], data['smoking']))
lebel = np.array(data['heart']).reshape(-1,1)
ones = np.ones((input.shape[0], 1))
input_new = np.hstack((ones, input))
X = np.array(input_new)
W = np.zeros((X.shape[1], 1))

#defing loss function
def loss_func(W,X,lebel):
  #W = W.reshape(-1,1)
  results = X @ W
  error_1 = results - lebel
  #print(error_1.shape[0])
  error_2 = np.square(error_1)
  sum = np.sum(error_2)
  #print(sum)
  error = (1/(2*error_1.shape[0]))*sum
  return error

#defining gradient descent
def gradient_descent(W,X,lebel,L):
    results = X @ W
    error_1 = results - lebel
    n = error_1.shape[0]
    M = error_1.T @ X
    M = M.reshape(-1,1)
    #L=0.001
    W = W - 2*L*M/n
    #print(W)
    return W

x3 =  np.linspace(0, 10, 40)
y3 =  np.linspace(0, 10, 40)
X_1, Y_1 = np.meshgrid(x3, y3)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.pause(1.0)

#learning part
for i in range(epochs):
    if  i < 15:
        c  = W[0,0]
        m1 = W[1,0]
        m2 = W[2,0]
        Z  = m1*X_1 + m2*Y_1 + c
        ax.plot_surface(X_1, Y_1, Z, cmap='cool', alpha=1)
        ax.scatter(data.biking, data.smoking, data.heart, color = "blue")
        ax.set_xlabel('Input Variable I')
        ax.set_ylabel('Input Variable II')
        ax.set_zlabel('Target Variable')
        plt.draw()
        plt.pause(1)
        plt.cla()
        print(m1,m2,c)        
    loss.append(loss_func(W,X,lebel))
    epo.append(i)
    #print(loss[i])
    W = gradient_descent(W,X,lebel,L)

#print the final value of m and c
print(W)


end = time.time()
print(f"Total runtime of the program is {end - start} seconds")

#shows final plot 
#plt.scatter(data.biking, data.smoking, data.heart, color = "blue")
ax.scatter(data.biking, data.smoking, data.heart, color = "red", marker="o")
Z  = W[1,0]*X_1 + W[2,0]*Y_1 + W[0,0]

ax.plot_surface(X_1, Y_1, Z, cmap='cool', alpha=0.8)

##ax.scatter(data.biking, data.smoking, data.heart, color = "blue")
ax.set_xlabel('Input Variable I')
ax.set_ylabel('Input Variable II')
ax.set_zlabel('Target Variable')
plt.draw()
plt.show()

