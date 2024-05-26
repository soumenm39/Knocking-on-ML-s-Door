# -*- coding: utf-8 -*-
"""
Created on Sun May 26 11:58:49 2024

@author: SOUMEN
"""
#basic linear regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data.csv')

def loss_func(m,c,points):
    error = 0
    for i in range (len(points)):
        x = points.iloc[i].abscissa 
        y = points.iloc[i].ordinate
        error = error + (y - (m*x + c))**2
    error/float(len(points))
    
def gradient_descent(new_m, new_c, points, L):
    m_gradient = 0
    c_gradient = 0
    
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].abscissa 
        y = points.iloc[i].ordinate
        
        m_gradient = m_gradient - (2/n)*x*(y - (new_m*x + new_c))
        c_gradient = c_gradient - (2/n)*(y - (new_m*x + new_c))
        
    m = new_m - m_gradient*L
    c = new_c - c_gradient*L    
    return m, c

m = 0
c = 0
L = 0.0001
epochs = 300

for i in range(epochs):
    if i < 10:
        print(f"Epoch: {i}")
        print(m,c)
        plt.scatter(data.abscissa, data.ordinate, color = "blue")
        plt.plot(list(range(10,100)), [m*x + c for x in range(10,100)], color="red")
        plt.pause(1.0)
        plt.close()
    m , c = gradient_descent(m , c, data, L)

plt.show()    


print(m,c)

plt.scatter(data.abscissa, data.ordinate, color = "blue")
plt.plot(list(range(10,100)), [m*x + c for x in range(10,100)], color="red")
plt.show()
