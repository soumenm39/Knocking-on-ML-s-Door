#Basic Multidimensional Linear Regression Model (2D) with two
#For special animated view use defult IDLE 

# %matplotlib widget  #only remove the #for jupyter note book

#import necessary library 
import pandas as pd  #for reading CSV file
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import matplotlib.pyplot as plt

#reading data from the 2D_regression_Data.csv
data = pd.read_csv('/kaggle/input/2d-regression-data/2D_regression_Data.csv')
data['biking'] = data.Input_Variable_I     
data['smoking'] = data.Input_Variable_I  
data['heart'] = data.Target_Variable  


#defing loss function
def loss_func(m1, m2, c, points):
    error = 0
    n = len(points)
    for i in range (len(points)):
        x1 = points.iloc[i].Input_Variable_I 
        x2 = points.iloc[i].Input_Variable_II
        y  = points.iloc[i].Target_Variable
        error = error + ((m1*x1 + m2*x2 + c) - y)**2
    error = error/(2*n)
    return error

#defining gradient descent
def gradient_descent(new_m1, new_m2, new_c, points, L):
    m1_gradient = 0
    m2_gradient = 0
    c_gradient = 0
    
    n = len(points)
    
    for i in range(n):
        x1 = points.iloc[i].Input_Variable_I 
        x2 = points.iloc[i].Input_Variable_II
        y  = points.iloc[i].Target_Variable
       
        m1_gradient = m1_gradient + x1*((new_m1*x1 +new_m2*x2 + new_c)-y)
        m2_gradient = m2_gradient + x2*((new_m1*x1 + new_m2*x2 + new_c)-y)
        c_gradient = c_gradient + ((new_m1*x1 + new_m2*x2 + new_c)-y)
        
    m1 = new_m1 - 2*(m1_gradient*L)/n
    m2 = new_m2 - 2*(m2_gradient*L)/n
    c = new_c - 2*(c_gradient*L)/n    
    return m1, m2,  c

#initial guess of m1, m2 and c
m1 = -0.005  
m2 = 0.1
c = 0.2
L = 0.0025  #learing rate 
epochs = 10000 #number of cycle

#Setting thing for animated plot for visulization
x3 =  np.linspace(0, 10, 40)
y3 =  np.linspace(0, 10, 40)
X, Y = np.meshgrid(x3, y3)
loss = []
epo  = []
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.pause(1.8)

#Loop calculate the values of m1, m2, and minimizing loss
for i in range(epochs):
   if i < 30:   #number of visulization step   
        Z  = m1*X + m2*Y + c
        ax.plot_surface(X, Y, Z, cmap='cool', alpha=0.8)
        ax.scatter(data.biking, data.smoking, data.heart, color = "blue")
        ax.set_xlabel('Input Variable I')
        ax.set_ylabel('Input Variable II')
        ax.set_zlabel('Target Variable')
        plt.draw()
        plt.pause(1.2)
        ax.cla()
   loss.append(loss_func(m1, m2, c, data))
   epo.append(i)
   m1, m2 , c = gradient_descent(m1 , m2, c, data, L)
   
#print the final value of m and c
print("The final value of m1, m2 and c of equation z = m1*x + m2*y + c be:")
print(m1,m2,c)

#shows final plot 
ax.scatter(data.biking, data.smoking, data.heart, color = "red", marker="o")
Z  = m1*X + m2*Y + c
ax.plot_surface(X, Y, Z, cmap='cool', alpha=0.8)
ax.set_xlabel('Input Variable I')
ax.set_ylabel('Input Variable II')
ax.set_zlabel('Target Variable')
plt.draw()
plt.show()

#uncommeting the following section will show the plot of Loss vs epochs 
#plt.close()
#figl = plt.figure(figsize = (5, 5))
#plt.plot(epo, loss, color="red")
#ax.set_xlabel('Epoch')
#ax.set_ylabel('Loss')
#plt.draw()
#plt.show()
