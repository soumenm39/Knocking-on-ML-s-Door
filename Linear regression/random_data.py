# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:21:37 2024

@author: SOUMEN
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

# generate x and y
x = np.linspace(10, 100, 101)
y = 1 + x*(-1)**2*np.random.uniform(low=1,high=3,size=len(x)) + x*np.random.random(size=len(x))

with open('data.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(('abscissa', 'ordinate'))
    writer.writerows(zip(x, y))


# plot the results
plt.figure(figsize = (10,8))
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')

plt.show()

