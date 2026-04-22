# 2D Linear Regression with Animated Visualization

## Overview

This project implements a 2D linear regression model using gradient descent to predict a target variable based on two input variables. The code uses **vectorized NumPy operations** for faster computation and cleaner implementation without explicit loops over training samples.

It visualizes the fitting process through animated 3D plots, showing how the regression plane evolves over iterations to fit the data. It also computes the loss at each epoch and optionally plots the loss vs. epochs graph.

This project is ideal for understanding the mechanics of linear regression, gradient descent, and vectorization in a multidimensional context, with a focus on practical visualization for educational purposes. 

The code was developed as part of my learning journey in machine learning and data analysis. The animated visualization reflects a hands-on approach to understanding ML algorithms, emphasizing practical implementation over automated solutions.


## Features

* Implements 2D linear regression using gradient descent
* Uses fully vectorized NumPy matrix operations
* Visualizes the fitting process with animated 3D plots
* Computes and tracks the loss function over epochs
* Optionally plots loss vs. epochs
* Uses a synthetic dataset (2D_regression_Data.csv) with two input variables and one target variable

### Requirements

### Software

* Python 3.6 or higher
* Jupyter Notebook (optional, for visualization)
* Python IDLE (recommended for animation)
* Git (for cloning the repository)

## Dataset

The code uses **2D_regression_Data.csv**, which contains the following columns:

* Input_Variable_I → First input feature
* Input_Variable_II → Second input feature
* Target_Variable → Target variable to predict

This is a synthetic dataset created for understanding regression learning and visualization.

---

## Installation

* Download the dataset **2D_regression_Data.csv**
* Place it in the same directory as the script
* Run the script in Python IDLE for the best animated visualization, or use Jupyter Notebook for plotting support

---

## Usage

* Open the script **2d_regression_vectorization.py**
* Ensure **2D_regression_Data.csv** is in the same directory
* Run the script

The script will:

* Load the dataset
* Perform vectorized gradient descent for training
* Visualize the first few iterations as animated 3D plots
* Display the final regression plane with data points
* Print the final learned values of weights and intercept
* Optionally show the loss vs. epochs graph


## Example Output

### Animated Plot

A 3D plot showing the regression plane evolving over iterations, with data points scattered and the final optimized plane.

### Final Parameters

The final values of m1, m2, and c for the equation:

$$
z = m_1x + m_2y + c
$$

Example:

```text id="s44s7r"
[[c]
 [m1]
 [m2]]
```

(Values will vary based on the dataset)


## Code Structure

* **Imports** → pandas, numpy, matplotlib, mpl_toolkits
* **Data Loading** → reads CSV and prepares input matrix
* **Loss Function** → computes mean squared error
* **Gradient Descent** → updates weights using vectorized operations
* **Visualization** → animated regression plane and final fitted surface
* **Loss Tracking** → stores loss values for each epoch


## Parameters

* Initial Weights → zeros initialization
* Learning Rate → 0.004
* Epochs → 2000 iterations
* Visualization → first few iterations are animated for clarity


## Notes

* The animation works best in Python IDLE due to matplotlib handling
* In Jupyter Notebook, plotting works but animation performance may vary
* Learning rate and epochs can be tuned for faster convergence or better accuracy


## Future Improvements

* Add feature normalization
* Compare with Scikit-learn implementation
* Add performance metrics like R² score


## Author

**Soumen Mondal**

Senior Research Fellow,
Saha Institute of Nuclear Physics (SINP),
DAE, Homi Bhabha National Institute

Email:
[sm39@iitbbs.ac.in](mailto:sm39@iitbbs.ac.in)
[soumen.mondal@saha.ac.in](mailto:soumen.mondal@saha.ac.in)

This project is part of my learning journey in ML.

## License

This project is licensed under the MIT License.


## Acknowledgments

* Synthetic dataset prepared for regression learning
* Inspired by my learning journey in machine learning
* Strongly motivated by understanding ML from first principles

