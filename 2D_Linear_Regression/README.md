**2D Linear Regression with Animated Visualization**

**Overview**
A 2D linear regression model using gradient descent, with animated 3D plots visualizing the fitting process. The dataset is synthetically generated based on a plane equation with added randomness. 

**Features**
_2D linear regression with gradient descent.

Animated 3D visualization of the regression plane fitting.

Synthetic dataset generation with noise.

Optional loss vs. epochs plot._

**Requirements**

Python 3.6+

Libraries: pandas, numpy, matplotlib, scipy


G_enerate the dataset (or use the provided 2D_regression_Data.csv):_
**
python generate_data.py**



**Animated 3D plot showing the regression plane fitting (first 30 iterations).**



_Final parameters (m1, m2, c) for z = m1*x + m2*y + c._



**Optional:** Loss vs. epochs plot (uncomment to enable).

**Files**





2d_linear_regression.py: Main script for regression and visualization.



generate_data.py: Generates 2D_regression_Data.csv using the plane z = 4 - 2x + 3y with noise.


2D_regression_Data.csv: Dataset with columns Input_Variable_I, Input_Variable_II, Target_Variable.

**Author**

Soumen Mondal
Senior Research Fellow, Saha Institute of Nuclear Physics
Email: sm39@iitbbs.ac.in
