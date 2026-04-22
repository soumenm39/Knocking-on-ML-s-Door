<img width="2880" height="1800" alt="loss_logistics_regression" src="https://github.com/user-attachments/assets/27e1701b-0e84-4bb4-8aa7-6683632fb1f6" /><img width="2880" height="1800" alt="loss_logistics_regression" src="https://github.com/user-attachments/assets/a7b0875b-a0b4-4f5f-b438-7f1b812f3f2f" /># Logistic Regression with Animated Decision Boundary


---

## Overview

A logistic regression model using gradient descent for binary classification, with animated decision boundary visualization during training. The dataset is used to classify whether a student will pass or fail based on two exam scores.


---

## Features

* Logistic regression using gradient descent
* Fully vectorized NumPy implementation
* Sigmoid activation function for binary classification
* Animated visualization of decision boundary fitting
* Feature normalization for stable convergence
* Loss vs. epochs plot
* Binary classification using exam score dataset



![Figure\_1](![Uploading 2D_Logistics_Regression - Made with Clipchamp.gif…])

---

## Requirements

Python 3.6+

Libraries: pandas, numpy, matplotlib, scipy

Install using:

```bash id="u1yzku"
pip install pandas numpy matplotlib scipy
```

---

## Dataset

The code uses **Exam_Data.csv** containing:

* GIExam → First input feature
* SubExam → Second input feature
* Output → Binary target variable

Where:

* 0 = Fail
* 1 = Pass

This dataset is used for binary classification and decision boundary visualization.

---

## Run the Code

```bash id="xw9z6g"
python Logistic_regression_vectorization.py
```

---

## Output

### Animated Decision Boundary Plot

A 2D plot showing how the decision boundary evolves during training to separate:

* Failed students (red)
* Passed students (green)

The blue line represents the classification boundary.

---

### Final Parameters

Final values of weights and intercept for:

[
c + m_1x + m_2y = 0
]

used as the final decision boundary.

---

### Loss vs. Epochs Plot

Shows how binary cross-entropy loss decreases over iterations.

---

## Files

* **Logistic_regression_vectorization.py** → Main script for logistic regression and visualization

* **Exam_Data.csv** → Dataset containing GIExam, SubExam, and Output columns

---

## Author

**Soumen Mondal**
Senior Research Fellow, Saha Institute of Nuclear Physics

Email: [sm39@iitbbs.ac.in](mailto:sm39@iitbbs.ac.in)

---
