# Build Your GPT — Solutions

## First Module: Math Foundations

### 1. Gradient Descent

#### Problem

Implement gradient descent to find the minimizer of the objective function:

$$
f(x) = x^2
$$

The derivative is:

$$
f'(x) = 2x
$$

The gradient descent update rule is:

$$
x = x - \alpha f'(x)
$$

where:

- $x$ = current value
- $\alpha$ = learning rate
- $f'(x)$ = derivative of the objective function
- `iterations` = number of update steps
- `init` = initial value of $x$

#### Solution

```python
class Solution:
    def get_minimizer(
        self,
        iterations: int,
        learning_rate: float,
        init: int
    ) -> float:

        # Objective function:
        # f(x) = x^2

        # Derivative:
        # f'(x) = 2x

        # Gradient descent update rule:
        # x = x - learning_rate * f'(x)

        x = init

        for i in range(iterations):
            derivative = 2 * x
            x = x - learning_rate * derivative

        # Round final answer to 5 decimal places
        return round(x, 5)

```
### 2. Activation Functions: Sigmoid and ReLU

#### Problem

Implement two commonly used activation functions:

1. **Sigmoid**
2. **ReLU (Rectified Linear Unit)**

The input `z` is a 1D NumPy array.

---

#### 1. Sigmoid

The sigmoid function is:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

It maps each value to a number between **0 and 1**.

#### 2.ReLU

The ReLU function is : 
$$
relu(z)=max(z,0)
$$

It ensures each value to be non negative .

#### Solution

```python
import numpy as np
from numpy.typing import NDArray


class Solution:

    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)

        ans = 1 / (1 + np.exp(-z))

        return np.round(ans, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise

        return np.maximum(z, 0)

```
### 3. Softmax

#### Problem

Implement the Softmax function for a 1D NumPy array of logits.

The Softmax function is:

$$
\text{softmax}(z_i)
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
$$

For numerical stability, subtract the maximum value of `z` before computing the exponential.

#### Solution

```python
import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)

        max_element = np.max(z)

        arrNew = z - max_element

        denom = 0

        for i in range(z.shape[0]):
            denom = denom + np.exp(arrNew[i])

        arr = np.zeros(z.shape[0])

        for i in range(z.shape[0]):
            nom = np.exp(arrNew[i])
            arr[i] = nom / denom

        return np.round(arr, 4)

```
### 4. Cross-Entropy Loss

#### Problem

Implement two commonly used loss functions:

1. **Binary Cross-Entropy (BCE)**
2. **Categorical Cross-Entropy (CCE)**

The predictions are clipped to avoid taking `log(0)`.

#### 1. Binary Cross-Entropy

The Binary Cross-Entropy formula is:

$$
BCE =
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i \log(\hat{y}_i)
+
(1-y_i)\log(1-\hat{y}_i)
\right]
$$

#### 2. Categorical Cross-Entropy

The Categorical Cross-Entropy formula is:

$$
CCE =
-\frac{1}{N}
\sum_{i=1}^{N}
\sum_{j=1}^{C}
y_{ij}\log(\hat{y}_{ij})
$$

#### Solution

```python
import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:

        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        bce = -(
            np.dot(
                y_true,
                np.log(y_pred_clipped)
            )
            +
            np.dot(
                (1 - y_true),
                np.log(1 - y_pred_clipped)
            )
        ) / y_true.shape[0]

        return round(bce, 4)


    def categorical_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:

        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        cce = 0

        for true_sample, pred_sample in zip(y_true, y_pred_clipped):

            temp_cce = -np.dot(
                true_sample,
                np.log(pred_sample)
            )

            cce += temp_cce

        cce = cce / y_true.shape[0]

        return np.round(cce, 4)

```
### 5. Linear Regression: Prediction and Mean Squared Error

#### Problem

Implement two functions:

1. **Model Prediction**
2. **Mean Squared Error (MSE)**

For model prediction:

- `X` has shape `(n, m)`
- `weights` has shape `(m,)`
- Predictions have shape `(n,)`

#### 1. Model Prediction

The prediction is calculated using:

$$
\hat{y} = Xw
$$

#### 2. Mean Squared Error

The Mean Squared Error is:

$$
MSE =
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
$$

#### Solution

```python
import numpy as np
from numpy.typing import NDArray


class Solution:

    def get_model_prediction(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places

        predictions = np.dot(X, weights)

        return np.round(predictions, 5)


    def get_error(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64]
    ) -> float:

        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places

        loss = model_prediction - ground_truth

        # The loss array is (n x 1) dimension,
        # so flatten it to get a 1D NumPy array

        loss_1D = loss.flatten()

        mse = np.dot(
            loss_1D,
            loss_1D
        ) / model_prediction.shape[0]

        return np.round(mse, 5)

```
### 6. Linear Regression: Gradient Descent

#### Problem

Implement three functions:

1. **Model Prediction**
2. **Derivative of the Loss**
3. **Model Training Using Gradient Descent**

For model prediction:

- `X` has shape `(n, m)`
- `weights` has shape `(m,)`
- Predictions have shape `(n,)`

The goal is to train a Linear Regression model by repeatedly updating its weights using Gradient Descent.

#### 1. Model Prediction

The prediction is calculated using:

$$
\hat{y} = Xw
$$

where:

- $X$ is the input feature matrix
- $w$ is the weight vector
- $\hat{y}$ is the model prediction

The matrix multiplication between `X` and `weights` produces the prediction for every training example.

#### 2. Derivative of the Loss

For Linear Regression, the Mean Squared Error is:

$$
MSE =
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
$$

To update a particular weight $w_j$, we need to calculate the derivative of the loss with respect to that weight.

The derivative is:

$$
\frac{\partial MSE}{\partial w_j}
=
\frac{-2}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)X_{ij}
$$

Using the dot product, this can be written as:

$$
\frac{\partial MSE}{\partial w_j}
=
\frac{-2}{N}
(y-\hat{y}) \cdot X_j
$$

where:

- $N$ is the number of training examples
- $y$ is the ground truth
- $\hat{y}$ is the model prediction
- $X_j$ is the $j$-th feature column
- $w_j$ is the weight corresponding to the $j$-th feature

#### 3. Gradient Descent

Gradient Descent updates the weights in the opposite direction of the gradient.

The update rule is:

$$
w_j =
w_j -
\alpha
\frac{\partial MSE}{\partial w_j}
$$

where:

- $w_j$ is the current weight
- $\alpha$ is the learning rate
- $\frac{\partial MSE}{\partial w_j}$ is the gradient

The learning rate used in this implementation is:

$$
\alpha = 0.01
$$

For every iteration:

1. Compute the model predictions using the current weights.
2. Calculate the derivative for each weight.
3. Update each weight using the Gradient Descent update rule.
4. Repeat this process for the specified number of iterations.
5. Round the final weights to 5 decimal places.

#### Training Process

Suppose we have:

- $n$ training examples
- $m$ features
- $m$ weights

For every iteration, the model first calculates:

$$
\hat{y} = Xw
$$

Then, for every weight $w_j$, it calculates:

$$
gradient_j =
\frac{-2}{N}
(y-\hat{y}) \cdot X_j
$$

The weight is then updated:

$$
w_j = w_j - \alpha \cdot gradient_j
$$

This process is repeated multiple times.

#### Key Idea

The main idea of Gradient Descent is to gradually change the weights so that the model's predictions become closer to the ground-truth values.

The overall process is:

$$
\boxed{
\text{Initial Weights}
\rightarrow
\text{Predictions}
\rightarrow
\text{Calculate Gradients}
\rightarrow
\text{Update Weights}
\rightarrow
\text{Repeat}
}
$$

As the number of iterations increases, the weights move toward values that minimize the Mean Squared Error.

#### Solution :
```python
import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N


    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)

        for i in range(num_iterations):

            y_pred=self.get_model_prediction(X,initial_weights)

            for j in range(len(initial_weights)):
                grad = self.get_derivative(y_pred,Y,X.shape[0],X,j)
                initial_weights[j]-=self.learning_rate * grad
            
        return np.round(initial_weights,5)       

```


#### Important Points

- `np.matmul(X, weights)` performs the matrix multiplication needed for prediction.
- `np.squeeze()` removes unnecessary dimensions from the prediction array.
- `X[:, desired_weight]` selects the feature column corresponding to a particular weight.
- `np.dot()` is used to calculate the dot product between the prediction error and the selected feature column.
- `X.shape[0]` gives the number of training examples, $N$.
- The learning rate controls how large each weight update is.
- A smaller learning rate makes smaller updates, while a larger learning rate makes larger updates.
- The weights are updated repeatedly until the specified number of iterations is completed.
- The final weights are rounded to 5 decimal places.