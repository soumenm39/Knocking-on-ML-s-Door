import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
import matplotlib.pyplot as plt
import random as rand
from mpl_toolkits.mplot3d import Axes3D

# uncomment for Jupyter note book
%matplotlib widget   

#plot of the equation of plane

def function(x, y):
    return 4 - 2*x + 3*y

#random introduced in the equation of the plane, more randomness can be introduced also
def data(x,y):
    return 4 - (2 + rand.random())*x + 3*(1 + rand.random())*y

#control the length and step of the dataset
x = np.linspace(0, 20, 100)
y = np.linspace(0, 20, 100)


# Step 2: Prepare output array
X, Y = np.meshgrid(x, y)

Z = function(X, Y)

fig = plt.figure() #figsize=(10, 8)
ax = fig.add_subplot(111, projection='3d')
#ax = Axes3D(fig)

#ploting the surface without randomness for reference only
ax.plot_surface(X, Y, Z, cmap='cool', alpha=0.8)

#creating target value and introducing a little more randomness 
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = data(X[i, j]+rand.random()*rand.random(), Y[i, j] + rand.random()*rand.random())

# Plotting data in a scatter plot
ax.scatter(X, Y, Z)

ax.set_title('Plot of Plane 2x−3y+z=4', fontsize=14)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_zlabel('z', fontsize=12)

plt.show()

# Flatten the arrays to 1D
x_flat = X.ravel()
y_flat = Y.ravel()
z_flat = Z.ravel()

# Stack X Y Z as columns
data = np.column_stack((x_flat, y_flat, z_flat))

# Save to CSV file
np.savetxt("2D_regression_Data.csv", data, header="Input_Variable_I Input_Variable_II Target_Variable", comments='', fmt="%.6f")
