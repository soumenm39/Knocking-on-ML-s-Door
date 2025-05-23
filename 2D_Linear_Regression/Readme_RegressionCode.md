2D Linear Regression with Animated Visualization

Overview

This project implements a 2D linear regression model using gradient descent to predict a target variable based on two input variables. The code visualizes the fitting process through animated 3D plots, showing how the regression plane evolves over iterations to fit the data. It also computes the loss at each epoch and optionally plots the loss vs. epochs graph. This project is ideal for understanding the mechanics of linear regression and gradient descent in a multidimensional context, with a focus on practical visualization for educational purposes.

The code was developed as part of my learning journey in machine learning and data analysis. The animated visualization reflects a hands-on approach to understanding ML algorithms, emphasizing practical implementation over automated solutions.

Features





Implements 2D linear regression using gradient descent.



Visualizes the fitting process with animated 3D plots, showing the regression plane's evolution.



Computes and tracks the loss function over epochs.



Optionally plots loss vs. epochs (commented out).



Uses a real dataset (2D_regression_Data.csv) with two input variables (Input_Variable_I, Input_Variable_II) and one target variable (Target_Variable).

Requirements

To run this code, you need the following:

Software





Python 3.6 or higher



Jupyter Notebook (optional, for %matplotlib widget functionality) or default Python IDLE for animation



Git (for cloning the repository)

Libraries

Install the required Python libraries using pip:

pip install pandas numpy matplotlib

Dataset





The code uses 2D_regression_Data.csv, which contains the following columns:





Input_Variable_I: First input feature (e.g., biking frequency).



Input_Variable_II: Second input feature (e.g., smoking frequency).



Target_Variable: Target variable to predict (e.g., heart health metric).



The dataset is sourced from Kaggle: 2D Regression Data. Download and place it in the same directory as the script.

Installation





Download the dataset 2D_regression_Data.csv


Run the script in Python IDLE for the best animated visualization, or use Jupyter Notebook with %matplotlib widget enabled.

Usage





Open the script 2d_linear_regression.py in your preferred environment (Python IDLE or Jupyter Notebook).



Ensure 2D_regression_Data.csv is in the same directory as the script.



Run the script:





In Python IDLE: Open and run the script directly to see the animated 3D plot.



In Jupyter Notebook: Uncomment the %matplotlib widget line at the top and run all cells.



The script will:





Load the dataset and rename columns (biking, smoking, heart) for clarity.



Perform gradient descent for 10,000 epochs with a learning rate of 0.0025.



Visualize the first 30 iterations as animated 3D plots, showing the regression plane fitting the data.



Display the final regression plane with the data points.



Print the final values of m1, m2, and c for the equation z = m1*x + m2*y + c.



Optionally, uncomment the loss vs. epochs plot section to visualize the loss convergence.

Example Output





Animated Plot: A 3D plot showing the regression plane evolving over the first 30 iterations, with data points scattered in blue (intermediate steps) and red (final plot).



Final Parameters:

The final value of m1, m2 and c of equation z = m1*x + m2*y + c be:
0.123456 0.789012 0.345678

(Values will vary based on the dataset and random initialization.)



Loss vs. Epochs Plot (if uncommented): A 2D plot showing the loss decreasing over 10,000 epochs.

Code Structure





Imports: Libraries for data handling (pandas), numerical operations (numpy), and plotting (matplotlib, mpl_toolkits).



Data Loading: Reads 2D_regression_Data.csv and renames columns for clarity.



Loss Function: Computes the mean squared error for the current parameters m1, m2, c.



Gradient Descent: Updates m1, m2, c using gradient descent with a learning rate of 0.0025.



Visualization:





Creates a 3D mesh grid for plotting the regression plane.



Animates the plane’s evolution over 30 iterations.



Plots the final fitted plane with data points.



Loss Tracking: Stores loss values for each epoch (optionally plotted).

Parameters





Initial Guess: m1 = -0.005, m2 = 0.1, c = 0.2



Learning Rate: L = 0.0025



Epochs: 10,000 iterations



Visualization Steps: First 30 iterations are animated for clarity

Notes


The animation works best in Python IDLE due to its handling of matplotlib animations. In Jupyter Notebook, %matplotlib widget is required for interactive plots, but performance may vary.



The learning rate and number of epochs can be tuned for faster convergence or better accuracy.

Future Improvements


Author

Soumen Mondal





Senior Research Fellow, Saha Institute of Nuclear Physics (SINP), DAE, Homi Bhabha National Institute



Email: sm39@iitbbs.ac.in | soumen.mondal@saha.ac.in



ResearchGate: Soumen Mondal

This project is part of my learning journey in ML

License

This project is licensed under the MIT License – see the LICENSE file for details.

Acknowledgments





Dataset source prepared using 2d_data_generation.py code: 2D_regression_Data.csv



Inspired by my NPTEL course Introduction to Machine Learning (IIT Madras), starting July 2025.
