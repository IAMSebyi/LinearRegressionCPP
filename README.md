# Linear Regression Model in C++

This project implements a linear regression model with multiple input features in C++ for educational purposes. It includes methods to train the model using gradient descent, predict outputs for given inputs, and calculate the model parameters.

## Features

- Train a linear regression model using gradient descent.
- Predict target values for new input data.
- Calculate and output model parameters (coefficients and intercept).
- Handle multiple input features and data points.

## Formulas

### Model Function

$$
f_{\vec{w},b}(\vec{x}) = \vec{w} * \vec{x} + b
$$

where $\vec{w}$ is the coefficients vector, $b$ is the intercept and $\vec{x}$ is the input features vector

### Squared Error Cost Function with L2 Regularization (Ridge)

$$
J(\vec{w}, b) = \frac{1}{2m}	\sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2 + \frac{\lambda}{2m}  \sum_{j=1}^{n} w_j^2
$$ 

where $m$ is the number of training examples, $\hat{y}^{(i)}$ is the predicted output target for the $i^{th}$ training data point, $y^{(i)}$ is the real target value of the $i^{th}$ training data point, $n$ is the number of input features, $\lambda$ is the regularization parameter and $w_j$ is the coefficient of the $j^{th}$ feature

### Gradient with respect to the coefficients vector parameter

$$
\frac{\delta}{\delta \vec{w}}J(\vec{w}, b) = \frac{1}{m}\sum_{i=1}^m(f_{\vec{w}, b}(\vec{x}^{(i)})-y^{(i)})*\vec{x}^{(i)} + \frac{\lambda}{m} * \vec{w}
$$ 

where $\vec{x}^{(i)}$ is the input features vector of the $i^{th}$ training data point

### Gradient with respect to the intercept parameter

$$
\frac{\delta}{\delta b}J(\vec{w}, b) = \frac{1}{m}\sum_{i=1}^m(f_{\vec{w}, b}(\vec{x}^{(i)})-y^{(i)})
$$

### Z-Score Normalization

$$
\vec{x}'=\frac{\vec{x}-\vec{\mu}}{\vec{\sigma}}
$$ 

where $\vec{x}$ is the original features vector, $\vec{\mu}$ is the mean of the features vectors and $\vec{\sigma}$ is the standard deviation of the features vectors

### Sample Standard Deviation

$$
\vec{\sigma} = \sqrt{\sum_{i=1}^{m}\frac{(\vec{x}^{(i)} - \vec{\mu})^2}{m-1}}
$$
