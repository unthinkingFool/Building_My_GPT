# Build Your GPT — Solutions

## Second Module: Build a Neural Net

### 1. Single Neuron

#### Problem

Implement a forward pass for a single neuron that supports two commonly used activation functions:

1. **Sigmoid**
2. **ReLU (Rectified Linear Unit)**

The neuron first performs a linear transformation:

$$
z = x \cdot w + b
$$

where:

- $x$ = input vector
- $w$ = weight vector
- $b$ = bias
- $z$ = pre-activation value

The activation function is then applied to $z$.

The function should support:

- `"sigmoid"`
- `"relu"`

The final output should be rounded to 5 decimal places.

#### 1. Linear Transformation

The first step of the forward pass is calculating the weighted sum:

$$
z = x \cdot w + b
$$

The dot product is:

$$
x \cdot w =
\sum_{i=1}^{n}x_iw_i
$$

Therefore:

$$
z =
\sum_{i=1}^{n}x_iw_i+b
$$

For example, if:

$$
x=[x_1,x_2,x_3]
$$

and:

$$
w=[w_1,w_2,w_3]
$$

then:

$$
z=x_1w_1+x_2w_2+x_3w_3+b
$$

The value $z$ is called the **pre-activation value** because the activation function has not been applied yet.

#### 2. Sigmoid Activation

If the activation function is `"sigmoid"`, apply the Sigmoid function:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

The Sigmoid function maps the input to a value between 0 and 1.

For example:

$$
\sigma(0)=0.5
$$

For a large positive value of $z$, the output approaches 1.

For a large negative value of $z$, the output approaches 0.

Sigmoid is commonly used when we want an output that can represent a probability.

#### 3. ReLU Activation

If the activation function is `"relu"`, apply the ReLU function:

$$
ReLU(z)=\max(0,z)
$$

This means:

$$
ReLU(z)=
\begin{cases}
z & \text{if } z>0\\
0 & \text{if } z\leq0
\end{cases}
$$

For example:

$$
ReLU(5)=5
$$

and:

$$
ReLU(-3)=0
$$

ReLU is one of the most commonly used activation functions in neural networks.

#### 4. Forward Pass

The complete forward pass can be represented as:

$$
x,w,b
\rightarrow
z=x\cdot w+b
\rightarrow
activation(z)
\rightarrow
y
$$

Therefore:

$$
y=activation(x\cdot w+b)
$$

If the activation is Sigmoid:

$$
y=
\frac{1}{1+e^{-(x\cdot w+b)}}
$$

If the activation is ReLU:

$$
y=
\max(0,x\cdot w+b)
$$

#### Solution

```python
import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        activation: str
    ) -> float:

        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)

        z = np.dot(x, w) + b

        if activation == "sigmoid":
            transformed = 1 / (1 + np.exp(-z))

        elif activation == "relu":
            transformed = max(z, 0)

        return float(np.round(transformed, 5))

```
### 2. Backward Pass: Computing Gradients

#### Problem

Implement the **backward pass** for a single neuron using the **sigmoid activation function**.

The neuron performs the following forward pass:

$$
z = x \cdot w + b
$$

Then the sigmoid activation is applied:

$$
\hat{y} = \sigma(z)
$$

where:

- $x$ = input vector
- $w$ = weight vector
- $b$ = bias
- $z$ = pre-activation value
- $\hat{y}$ = predicted output
- $y_{true}$ = actual target value

The loss function is:

$$
L = \frac{1}{2}(\hat{y} - y_{true})^2
$$

The goal of the backward pass is to calculate:

1. The gradient of the loss with respect to each weight:

$$
\frac{\partial L}{\partial w}
$$

2. The gradient of the loss with respect to the bias:

$$
\frac{\partial L}{\partial b}
$$

---

#### 1. Forward Pass

First, calculate the pre-activation:

$$
z = x \cdot w + b
$$

Then apply the sigmoid function:

$$
\hat{y} = \frac{1}{1 + e^{-z}}
$$

The prediction $\hat{y}$ is then used to calculate the loss:

$$
L = \frac{1}{2}(\hat{y} - y_{true})^2
$$

---

#### 2. Sigmoid Derivative

The derivative of the sigmoid function is:

$$
\frac{d\hat{y}}{dz}
=
\hat{y}(1-\hat{y})
$$

This tells us how much the prediction changes when the pre-activation $z$ changes.

---

#### 3. Gradient of the Loss with Respect to the Prediction

Starting with:

$$
L = \frac{1}{2}(\hat{y} - y_{true})^2
$$

Differentiate with respect to $\hat{y}$:

$$
\frac{\partial L}{\partial \hat{y}}
=
\hat{y} - y_{true}
$$

---

#### 4. Gradient with Respect to the Weights

Using the **chain rule**:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial z}
\frac{\partial z}{\partial w}
$$

We know:

$$
\frac{\partial L}{\partial \hat{y}}
=
\hat{y} - y_{true}
$$

and:

$$
\frac{\partial \hat{y}}{\partial z}
=
\hat{y}(1-\hat{y})
$$

Since:

$$
z = x \cdot w + b
$$

we have:

$$
\frac{\partial z}{\partial w}=x
$$

Therefore:

$$
\boxed{
\frac{\partial L}{\partial w}
=
(\hat{y}-y_{true})\hat{y}(1-\hat{y})x
}
$$

Because $x$ is a vector, the weight gradient is also a vector.

---

#### 5. Gradient with Respect to the Bias

Again using the chain rule:

$$
\frac{\partial L}{\partial b}
=
\frac{\partial L}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial z}
\frac{\partial z}{\partial b}
$$

Since:

$$
z = x \cdot w + b
$$

we have:

$$
\frac{\partial z}{\partial b}=1
$$

Therefore:

$$
\boxed{
\frac{\partial L}{\partial b}
=
(\hat{y}-y_{true})\hat{y}(1-\hat{y})
}
$$

---

#### 6. Backpropagation Chain

The complete forward computation is:

$$
x,w,b
\rightarrow
z
\rightarrow
\hat{y}
\rightarrow
L
$$

During backpropagation, we move in the opposite direction:

$$
L
\rightarrow
\hat{y}
\rightarrow
z
\rightarrow
w,b
$$

The final gradients are:

$$
\boxed{
\frac{\partial L}{\partial w}
=
(\hat{y}-y_{true})\hat{y}(1-\hat{y})x
}
$$

and:

$$
\boxed{
\frac{\partial L}{\partial b}
=
(\hat{y}-y_{true})\hat{y}(1-\hat{y})
}
$$

---

#### Solution

```python
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        y_true: float
    ) -> Tuple[NDArray[np.float64], float]:

        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward:
        # z = dot(x, w) + b
        # y_hat = sigmoid(z)
        #
        # Loss:
        # L = 0.5 * (y_hat - y_true)^2
        #
        # Return:
        # (dL_dw rounded to 5 decimals,
        #  dL_db rounded to 5 decimals)

        z = np.dot(x, w) + b

        y_hat = 1 / (1 + np.exp(-z))

        loss = 0.5 * (y_hat - y_true) ** 2

        # y_hat = sigmoid(xw + b)

        dl_dw = (y_hat - y_true) * y_hat * (1 - y_hat) * x

        dl_db = (y_hat - y_true) * y_hat * (1 - y_hat)

        return np.round(dl_dw, 5), np.round(dl_db, 5)

```


### 3. Forward and Backward Pass: Two-Layer Neural Network

#### Problem

Implement both the **forward pass** and **backward pass** for a simple two-layer neural network.

The architecture is:

$$
x
\rightarrow
\text{Linear}(W_1,b_1)
\rightarrow
\text{ReLU}
\rightarrow
\text{Linear}(W_2,b_2)
\rightarrow
\hat{y}
$$

The loss function is **Mean Squared Error (MSE)**:

$$
L =
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{y}_i-y_i)^2
$$

The function should return:

- `loss`
- `dW1`
- `db1`
- `dW2`
- `db2`

---

# 1. Understand the Network First

Before looking at the code, we need to understand the shape of every object.

Suppose:

- Input size = $I$
- Hidden layer size = $H$
- Output size = $O$

We use the following convention:

$$
W_1=(H,I)
$$

$$
W_2=(O,H)
$$

So the network looks like:

```text
Input
  x
  |
  | W1
  ↓
z1
  |
  | ReLU
  ↓
a1
  |
  | W2
  ↓
z2
  |
  ↓
y_hat
```

The complete flow is:

```text
x
 ↓
z1 = W1x + b1
 ↓
a1 = ReLU(z1)
 ↓
z2 = W2a1 + b2
 ↓
y_hat = z2
 ↓
Loss
```

---

# 2. Dimensions of Everything

This is one of the most important parts of the problem.

Assume:

```text
input size  = I
hidden size = H
output size = O
```

Then:

| Variable | Shape | Meaning |
|---|---|---|
| `x` | `(I,)` | Input |
| `W1` | `(H, I)` | First-layer weights |
| `b1` | `(H,)` | First-layer bias |
| `z1` | `(H,)` | First-layer output before activation |
| `a1` | `(H,)` | Hidden-layer activation |
| `W2` | `(O, H)` | Second-layer weights |
| `b2` | `(O,)` | Second-layer bias |
| `z2` | `(O,)` | Second-layer output |
| `y_pred` | `(O,)` | Prediction |
| `y_true` | `(O,)` | Actual target |

For example, if:

```text
input size  = 2
hidden size = 3
output size = 1
```

then:

```text
x      → (2,)
W1     → (3, 2)
b1     → (3,)
z1     → (3,)
a1     → (3,)

W2     → (1, 3)
b2     → (1,)
z2     → (1,)
y_pred → (1,)
```

---

# 3. Why Is W1 `(hidden_size, input_size)`?

The first layer performs:

$$
z_1=W_1x+b_1
$$

Suppose:

$$
W_1:(H,I)
$$

and:

$$
x:(I,)
$$

Then:

$$
(H,I)\times(I,)
\rightarrow
(H,)
$$

The inner dimensions must match.

```text
        W1             x

      H × I           I
       ↓              ↓

      [  ]            [ ]
      [  ]      ×     [ ]
      [  ]

       H × I           I

              ↓

             H
```

The `I` from `W1` matches the `I` from `x`.

The result has `H` values, one for each hidden neuron.

Therefore:

$$
\boxed{
W_1:(H,I)
}
$$

---

# 4. First Forward Pass

The first operation is:

$$
z_1=W_1x+b_1
$$

In Python:

```python
z1 = np.dot(w1, x) + b1
```

Notice the order:

```text
w1, x
```

not:

```text
x, w1
```

because `W1` is:

```text
(H, I)
```

and `x` is:

```text
(I,)
```

So:

```text
(H, I) × (I,)
      ↓
     (H,)
```

This gives one value for every hidden neuron.

---

# 5. ReLU Activation

After calculating `z1`, we apply ReLU.

The ReLU function is:

$$
ReLU(x)=\max(0,x)
$$

Therefore:

```python
a1 = np.maximum(z1, 0)
```

The important thing is that ReLU does **not change the dimensions**.

If:

```text
z1 → (H,)
```

then:

```text
a1 → (H,)
```

For example:

```text
z1 = [-2, 3, -1, 5]

ReLU

a1 = [0, 3, 0, 5]
```

---

# 6. Why Is W2 `(output_size, hidden_size)`?

The second layer takes the hidden-layer output as its input.

We have:

$$
a_1:(H,)
$$

The second layer calculates:

$$
z_2=W_2a_1+b_2
$$

Therefore, if:

$$
W_2:(O,H)
$$

then:

$$
(O,H)\times(H,)
\rightarrow
(O,)
$$

Again, the inner dimensions must match.

```text
       W2              a1

      O × H             H
       ↓                ↓

      [  ]              [ ]
      [  ]       ×      [ ]
      [  ]              [ ]

      O × H             H

               ↓

               O
```

Therefore:

$$
\boxed{
W_2:(O,H)
}
$$

---

# 7. Second Forward Pass

Mathematically:

$$
z_2=W_2a_1+b_2
$$

The code is:

```python
z2 = np.dot(a1, w2.T) + b2
```

Why is the code written as:

```python
np.dot(a1, w2.T)
```

instead of:

```python
np.dot(w2, a1)
```

Both represent the same mathematical operation here.

We have:

```text
a1 → (H,)
W2 → (O,H)
```

and:

```text
W2.T → (H,O)
```

Therefore:

```text
a1 × W2.T

(H,) × (H,O)
      ↓
     (O,)
```

This is equivalent to:

```text
W2 × a1

(O,H) × (H,)
      ↓
     (O,)
```

So:

```python
np.dot(a1, w2.T)
```

and:

```python
np.dot(w2, a1)
```

produce the same result for this 1D vector representation.

---

# 8. Prediction

There is no activation function after the second linear layer.

Therefore:

$$
\hat{y}=z_2
$$

In Python:

```python
y_pred = z2
```

---

# 9. MSE Loss

The Mean Squared Error is:

$$
L=
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{y}_i-y_i)^2
$$

In Python:

```python
loss = np.mean((y_pred - y_true) ** 2)
```

The loss is a scalar.

For example:

```text
y_pred = [1.5]
y_true = [1.0]

error = 0.5

loss = 0.5²
     = 0.25
```

---

# 10. Now Start the Backward Pass

The forward pass goes from left to right:

```text
x
 ↓
z1
 ↓
a1
 ↓
z2
 ↓
y_pred
 ↓
Loss
```

Backpropagation goes in the **opposite direction**:

```text
Loss
 ↓
y_pred
 ↓
z2
 ↓
a1
 ↓
z1
 ↓
x
```

At each step, we use the **chain rule**.

This is why the backward pass must be performed in this order.

---

# 11. Gradient of MSE

The loss is:

$$
L=
\frac{1}{N}
\sum
(\hat{y}-y)^2
$$

The derivative is:

$$
\frac{\partial L}{\partial\hat{y}}
=
\frac{2(\hat{y}-y)}{N}
$$

Since:

$$
\hat{y}=z_2
$$

we have:

$$
dz_2=
\frac{2(\hat{y}-y)}{N}
$$

In Python:

```python
dz2 = 2 * (y_pred - y_true) / y_true.size
```

The shape is:

```text
dz2 → (O,)
```

---

# 12. Gradient of W2

The second layer is:

$$
z_2=W_2a_1+b_2
$$

We want:

$$
\frac{\partial L}{\partial W_2}
$$

For an individual weight:

$$
\frac{\partial L}{\partial W_{2,ij}}
=
\frac{\partial L}{\partial z_{2,i}}
\frac{\partial z_{2,i}}
{\partial W_{2,ij}}
$$

Since:

$$
z_{2,i}
=
\sum_j W_{2,ij}a_{1,j}+b_{2,i}
$$

we get:

$$
\frac{\partial z_{2,i}}
{\partial W_{2,ij}}
=
a_{1,j}
$$

Therefore:

$$
\frac{\partial L}{\partial W_{2,ij}}
=
dz_{2,i}a_{1,j}
$$

This is exactly an outer product:

$$
dW_2=dz_2\otimes a_1
$$

In NumPy:

```python
dw2 = np.outer(dz2, a1)
```

---

# 13. Why `np.outer(dz2, a1)`?

Suppose:

```text
dz2 → (O,)
a1  → (H,)
```

The outer product creates:

```text
(O, H)
```

which is exactly the shape of:

```text
W2 → (O, H)
```

For example:

```text
dz2 = [g]

a1 = [a, b, c]
```

Then:

```text
np.outer(dz2, a1)

       a       b       c
     ----------------------
g  |  ga      gb      gc
```

The result is:

```text
(1, 3)
```

which matches:

```text
W2 → (1, 3)
```

Therefore:

```python
dw2 = np.outer(dz2, a1)
```

---

# 14. Why Can't We Reverse the Outer Product?

Suppose we write:

```python
np.outer(a1, dz2)
```

Then:

```text
a1  → (H,)
dz2 → (O,)
```

The result becomes:

```text
(H, O)
```

But `W2` has shape:

```text
(O, H)
```

So the gradient would be transposed.

For example:

```text
Correct:

dW2 → (1, 3)

[g*a  g*b  g*c]
```

Wrong order:

```text
(3, 1)

[g*a]
[g*b]
[g*c]
```

Therefore:

$$
\boxed{
dW_2=dz_2\otimes a_1}
$$

The gradient must have the same shape as `W2`.

---

# 15. Gradient of b2

The second layer is:

$$
z_2=W_2a_1+b_2
$$

The derivative with respect to the bias is:

$$
\frac{\partial z_2}{\partial b_2}=1
$$

Therefore:

$$
db_2=dz_2
$$

In Python:

```python
db2 = dz2
```

The shape remains:

```text
db2 → (O,)
```

---

# 16. Backpropagate to a1

Now we need to send the gradient backward through the second linear layer.

We have:

$$
z_2=W_2a_1+b_2
$$

We want:

$$
\frac{\partial L}{\partial a_1}
$$

Using the chain rule:

$$
da_1=W_2^Tdz_2
$$

In mathematical matrix notation:

```text
W2.T → (H, O)

dz2 → (O,)

(H, O) × (O,)
        ↓
       (H,)
```

Therefore:

$$
da_1:(H,)
$$

The code is:

```python
da1 = np.dot(dz2, w2)
```

Because `dz2` and `w2` are represented as:

```text
dz2 → (O,)
W2  → (O,H)
```

NumPy's 1D dot operation gives:

```text
(O,) × (O,H)
      ↓
     (H,)
```

This is equivalent to:

```python
da1 = np.dot(w2.T, dz2)
```

which follows the mathematical matrix notation more directly.

---

# 17. Why Does the Gradient Go Through W2.T?

This is an important rule.

Forward:

$$
z_2=W_2a_1
$$

where:

```text
W2 → (O,H)
a1 → (H,)
```

produces:

```text
z2 → (O,)
```

Backward:

```text
dz2 → (O,)
```

needs to become:

```text
da1 → (H,)
```

Therefore, we need:

```text
W2.T → (H,O)

(H,O) × (O,)
       ↓
      (H,)
```

So the transpose is necessary to reverse the direction of the forward linear transformation.

General rule:

> During the backward pass through a matrix multiplication, the weight matrix is transposed when propagating the gradient back to the input.

---

# 18. Backpropagate Through ReLU

We have:

$$
a_1=ReLU(z_1)
$$

The derivative of ReLU is:

$$
ReLU'(z)=
\begin{cases}
1 & z>0\\
0 & z\leq0
\end{cases}
$$

Therefore:

$$
dz_1
=
da_1\odot ReLU'(z_1)
$$

In Python:

```python
dz1 = da1 * (z1 > 0)
```

Here:

```python
(z1 > 0)
```

creates a Boolean array.

For example:

```text
z1 = [-2, 3, -1, 5]

z1 > 0

[False, True, False, True]
```

When multiplied:

```text
da1 = [2, 4, 6, 8]

ReLU gradient:

[0, 1, 0, 1]

dz1:

[0, 4, 0, 8]
```

So ReLU blocks the gradient for neurons whose input was not positive.

---

# 19. Gradient of W1

The first layer is:

$$
z_1=W_1x+b_1
$$

We want:

$$
dW_1
=
\frac{\partial L}{\partial W_1}
$$

For an individual weight:

$$
\frac{\partial L}{\partial W_{1,ij}}
=
\frac{\partial L}{\partial z_{1,i}}
\frac{\partial z_{1,i}}
{\partial W_{1,ij}}
$$

Since:

$$
z_{1,i}
=
\sum_j W_{1,ij}x_j+b_{1,i}
$$

we have:

$$
\frac{\partial z_{1,i}}
{\partial W_{1,ij}}
=
x_j
$$

Therefore:

$$
\frac{\partial L}{\partial W_{1,ij}}
=
dz_{1,i}x_j
$$

This gives:

$$
dW_1=dz_1\otimes x
$$

In Python:

```python
dw1 = np.outer(dz1, x)
```

---

# 20. Why `np.outer(dz1, x)`?

Suppose:

```text
dz1 → (H,)
x   → (I,)
```

Then:

```text
np.outer(dz1, x)

→ (H, I)
```

which exactly matches:

```text
W1 → (H, I)
```

For example:

```text
dz1 = [g1, g2, g3]

x = [x1, x2]
```

The outer product is:

```text
          x1        x2
       ------------------
g1  |   g1*x1     g1*x2
g2  |   g2*x1     g2*x2
g3  |   g3*x1     g3*x2
```

Therefore:

```text
dW1 → (3, 2)
```

which matches:

```text
W1 → (3, 2)
```

---

# 21. Why Not `np.outer(x, dz1)`?

If we reverse the order:

```python
np.outer(x, dz1)
```

we get:

```text
(I, H)
```

instead of:

```text
(H, I)
```

For example:

```text
Correct:

dW1 → (3, 2)

[g1*x1  g1*x2]
[g2*x1  g2*x2]
[g3*x1  g3*x2]
```

Wrong order:

```text
(2, 3)

[g1*x1  g2*x1  g3*x1]
[g1*x2  g2*x2  g3*x2]
```

The values are the same, but the matrix is transposed.

Since the gradient must match the shape of `W1`, the correct expression is:

```python
dw1 = np.outer(dz1, x)
```

Therefore:

$$
\boxed{
dW_1=dz_1\otimes x}
$$

---

# 22. Gradient of b1

The first layer is:

$$
z_1=W_1x+b_1
$$

The derivative with respect to the bias is:

$$
\frac{\partial z_1}{\partial b_1}=1
$$

Therefore:

$$
db_1=dz_1
$$

In Python:

```python
db1 = dz1
```

The shape is:

```text
db1 → (H,)
```

---

# 23. Complete Forward Pass

The complete forward pass is:

```text
                    Forward

x
│
│  W1 @ x + b1
↓
z1
│
│  ReLU
↓
a1
│
│  W2 @ a1 + b2
↓
z2
│
↓
y_pred
│
│  MSE
↓
Loss
```

Mathematically:

$$
\boxed{
z_1=W_1x+b_1
}
$$

$$
\boxed{
a_1=ReLU(z_1)
}
$$

$$
\boxed{
z_2=W_2a_1+b_2
}
$$

$$
\boxed{
\hat{y}=z_2
}
$$

$$
\boxed{
L=\frac{1}{N}\sum(\hat{y}-y)^2
}
$$

---

# 24. Complete Backward Pass

The backward pass reverses the forward operations:

```text
                    Backward

Loss
│
│  MSE derivative
↓
dz2
│
├──────────────→ dW2 = outer(dz2, a1)
│
├──────────────→ db2 = dz2
│
│
│  W2.T @ dz2
↓
da1
│
│  ReLU derivative
↓
dz1
│
├──────────────→ dW1 = outer(dz1, x)
│
└──────────────→ db1 = dz1
```

The complete equations are:

$$
\boxed{
dz_2=
\frac{2(\hat{y}-y)}{N}
}
$$

$$
\boxed{
dW_2=dz_2\otimes a_1
}
$$

$$
\boxed{
db_2=dz_2
}
$$

$$
\boxed{
da_1=W_2^Tdz_2
}
$$

$$
\boxed{
dz_1=da_1\odot ReLU'(z_1)
}
$$

$$
\boxed{
dW_1=dz_1\otimes x
}
$$

$$
\boxed{
db_1=dz_1
}
$$

---

# 25. Complete Dimension Flow

For:

```text
Input size  = I
Hidden size = H
Output size = O
```

the forward pass is:

```text
x
(I,)

      W1
(H,I)

(H,I) × (I,)
       ↓
z1
(H,)

      ReLU
       ↓
a1
(H,)

      W2
(O,H)

(O,H) × (H,)
       ↓
z2
(O,)

       ↓
y_pred
(O,)
```

The backward pass is:

```text
dz2
(O,)

       ↓

dW2 = outer(dz2, a1)

(O,) × (H,)
       ↓
(O,H)

       ↓

da1 = W2.T @ dz2

(H,O) × (O,)
       ↓
(H,)

       ↓

dz1
(H,)

       ↓

dW1 = outer(dz1, x)

(H,) × (I,)
       ↓
(H,I)
```

Notice how the gradients always return to the same shape as their corresponding parameters.

---

# 26. General Rule for Weight Gradients

For a linear layer:

$$
z=Wx+b
$$

the weight matrix has shape:

$$
W=(output,input)
$$

The gradient is:

$$
\boxed{
dW=dz\otimes x
}
$$

where:

- `dz` = gradient coming into the layer
- `x` = input that entered the layer

Therefore:

```text
Gradient × Input
```

using an outer product gives:

```text
(output,) × (input,)
             ↓
(output, input)
```

which matches the weight matrix.

For the first layer:

$$
dW_1=dz_1\otimes x
$$

For the second layer:

$$
dW_2=dz_2\otimes a_1
$$

---

# 27. Why Backpropagation Uses the Reverse Order

Forward propagation calculates:

```text
x
 ↓
z1
 ↓
a1
 ↓
z2
 ↓
Loss
```

To calculate the gradient of the loss with respect to something earlier in the network, we need the gradient from the operation after it.

For example, to calculate:

$$
\frac{\partial L}{\partial W_1}
$$

we need:

$$
\frac{\partial L}{\partial z_1}
$$

To calculate that, we need:

$$
\frac{\partial L}{\partial a_1}
$$

To calculate that, we need:

$$
\frac{\partial L}{\partial z_2}
$$

Therefore, the chain naturally forces us to move backward:

```text
L
↓
z2
↓
a1
↓
z1
↓
W1
```

This is why backpropagation is performed from the output layer toward the input layer.

---

# 28. Complete Solution

```python
import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float]
    ) -> dict:

        # Convert lists to NumPy arrays
        x = np.array(x)
        w1 = np.array(W1)
        b1 = np.array(b1)
        w2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        # ============================================================
        # FORWARD PASS
        # ============================================================

        # First linear layer
        # W1: (hidden, input)
        # x : (input,)
        # Result: (hidden,)
        z1 = np.dot(w1, x) + b1

        # ReLU activation
        # z1: (hidden,)
        # a1: (hidden,)
        a1 = np.maximum(z1, 0)

        # Second linear layer
        # a1: (hidden,)
        # W2: (output, hidden)
        # W2.T: (hidden, output)
        # Result: (output,)
        z2 = np.dot(a1, w2.T) + b2

        # No activation after the output layer
        y_pred = z2

        # MSE loss
        loss = np.mean((y_pred - y_true) ** 2)

        # ============================================================
        # BACKWARD PASS
        # ============================================================

        # Gradient of MSE with respect to prediction
        #
        # dL/dy_pred = 2 * (y_pred - y_true) / N
        dz2 = 2 * (y_pred - y_true) / y_true.size

        # Gradient of W2
        #
        # dz2: (output,)
        # a1 : (hidden,)
        #
        # outer product:
        # (output,) × (hidden,)
        #        -> (output, hidden)
        #
        # This matches W2.
        dw2 = np.outer(dz2, a1)

        # Gradient of b2
        #
        # db2 has the same shape as b2.
        db2 = dz2

        # Gradient flowing backward to a1
        #
        # Mathematically:
        #
        # da1 = W2.T @ dz2
        #
        # With NumPy's 1D representation, this is equivalent to:
        da1 = np.dot(dz2, w2)

        # Gradient through ReLU
        #
        # ReLU'(z) = 1 when z > 0
        # ReLU'(z) = 0 otherwise
        dz1 = da1 * (z1 > 0)

        # Gradient of W1
        #
        # dz1: (hidden,)
        # x  : (input,)
        #
        # outer product:
        # (hidden,) × (input,)
        #        -> (hidden, input)
        #
        # This matches W1.
        dw1 = np.outer(dz1, x)

        # Gradient of b1
        #
        # db1 has the same shape as b1.
        db1 = dz1

        return {
            "loss": float(np.round(loss, 4)),
            "dW1": np.round(dw1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dw2, 4).tolist(),
            "db2": np.round(db2, 4).tolist()
        }
```

---

# 29. Important Things to Remember

### 1. Weight matrix dimensions

For a layer:

$$
z=Wx+b
$$

use:

$$
\boxed{
W=(output,input)
}
$$

Therefore:

```text
W1 → (hidden, input)

W2 → (output, hidden)
```

---

### 2. Forward multiplication

Always make sure the inner dimensions match.

For the first layer:

```text
W1 @ x

(hidden, input) × (input,)
                  ↓
                (hidden,)
```

For the second layer:

```text
W2 @ a1

(output, hidden) × (hidden,)
                    ↓
                  (output,)
```

---

### 3. Backpropagation through a linear layer

For:

$$
z=Wx+b
$$

the gradient flowing backward to the input is:

$$
\boxed{
dx=W^Tdz
}
$$

The transpose appears because we are reversing the direction of the forward matrix multiplication.

---

### 4. Weight gradient

For:

$$
z=Wx+b
$$

the weight gradient is:

$$
\boxed{
dW=dz\otimes x
}
$$

So the order is:

```text
gradient first
input second
```

For the first layer:

```python
np.outer(dz1, x)
```

For the second layer:

```python
np.outer(dz2, a1)
```

---

### 5. Bias gradient

For:

$$
z=Wx+b
$$

the bias gradient is simply:

$$
\boxed{
db=dz
}
$$

because:

$$
\frac{\partial z}{\partial b}=1
$$

---

### 6. ReLU backward pass

For ReLU:

$$
ReLU(z)=\max(0,z)
$$

the derivative is:

$$
ReLU'(z)=
\begin{cases}
1 & z>0\\
0 & z\leq0
\end{cases}
$$

Therefore:

```python
dz = da * (z > 0)
```

---

# 30. The Most Important Mental Model

Think of every linear layer as:

```text
Forward:

input
  ↓
Wx + b
  ↓
output
```

During backward propagation:

```text
Backward:

gradient from output
        ↓
       dz
       / \
      /   \
     ↓     ↓
    dW     db

and

gradient to input:

dx = W.T @ dz
```

The three most important equations are:

### Weight gradient

$$
\boxed{
dW=dz\otimes input
}
$$

### Bias gradient

$$
\boxed{
db=dz
}
$$

### Gradient to previous layer

$$
\boxed{
dx=W^Tdz
}
$$

These three equations are extremely important for understanding backpropagation in neural networks.

---

# Quick Cheat Sheet

For a linear layer:

$$
z=Wx+b
$$

where:

```text
W → (output, input)
x → (input,)
z → (output,)
```

Forward:

$$
\boxed{z=Wx+b}
$$

Backward:

$$
\boxed{dz=\text{gradient from next layer}}
$$

$$
\boxed{dW=dz\otimes x}
$$

$$
\boxed{db=dz}
$$

$$
\boxed{dx=W^Tdz}
$$

Remember:

```text
             FORWARD
                ↓

       W @ input + bias
                ↓
             output


             BACKWARD
                ↓

       gradient from output
                ↓
              dz
             /  \
            /    \
          dW      db

       dx = W.T @ dz
```

The dimensions should always guide you.

If the gradient has the wrong shape, check:

1. The shape of `W`
2. The shape of the input
3. The order of the multiplication
4. Whether a transpose is required
5. The order of the arguments in `np.outer()`

### 4. Forward Pass — Multi-Layer Neural Network

#### Problem

Implement the **forward pass** for a multi-layer neural network.

The network consists of:

- An input vector `x`
- Multiple hidden layers
- An output layer
- A weight matrix and bias vector for each layer

The forward pass for each layer is:

$$
z = aW + b
$$

where:

- $a$ is the input/activation from the previous layer
- $W$ is the weight matrix
- $b$ is the bias vector
- $z$ is the output before activation

For every **hidden layer**, apply the ReLU activation function:

$$
\text{ReLU}(z) = \max(0,z)
$$

The **output layer should have no activation function**.

Therefore, for a network with two layers:

$$
z_1 = xW_1 + b_1
$$

$$
a_1 = \text{ReLU}(z_1)
$$

and then:

$$
z_2 = a_1W_2 + b_2
$$

Since the second layer is the output layer:

$$
y = z_2
$$

---

#### Solution

```python
import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(
        self,
        x: NDArray[np.float64],
        weights: List[NDArray[np.float64]],
        biases: List[NDArray[np.float64]]
    ) -> NDArray[np.float64]:

        num_layer = len(weights)

        ans = x

        for i in range(num_layer):

            # Linear transformation
            y_hat = np.dot(ans, weights[i]) + biases[i]

            # Apply ReLU only to hidden layers
            if i < num_layer - 1:
                ans = np.maximum(y_hat, 0)
            else:
                # No activation on the output layer
                ans = y_hat

        return np.round(ans, 5)
```

---


### 5. Xavier and Kaiming Initialization

#### Problem

Implement two widely used weight-initialization schemes and a helper that checks how well an initialization keeps activation statistics stable across many layers:

1. **Xavier / Glorot initialization** — best suited for **symmetric activations** such as `tanh` or `sigmoid`.
2. **Kaiming / He initialization** — best suited for **ReLU-family activations**.
3. **`check_activations`** — builds a small deep network with a chosen initialization scheme and reports how the standard deviation of the activations evolves layer by layer.

---

#### 1. Why Weight Initialization Matters

Before touching the formulas, it helps to understand **why** we can't just initialize weights randomly with, say, `std = 1`.

A deep network is a repeated composition of:

$$
z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}
$$

$$
a^{(l)} = f(z^{(l)})
$$

If the variance of the activations **grows** with every layer, the values eventually explode (numbers become huge, gradients become huge → **exploding gradients**).

If the variance **shrinks** with every layer, the values eventually collapse toward zero (gradients become tiny → **vanishing gradients**).

The goal of a good initialization scheme is:

$$
\boxed{
\text{Var}(a^{(l)}) \approx \text{Var}(a^{(l-1)})
}
$$

In other words, **keep the variance of the signal roughly constant as it flows through the network**, both on the forward pass (activations) and on the backward pass (gradients).

This is exactly the problem that Xavier initialization (Glorot & Bengio, 2010) and Kaiming initialization (He et al., 2015) were designed to solve.

---

#### 2. Variance of a Linear Layer — The Core Derivation

Consider a single pre-activation output of a linear layer:

$$
z = \sum_{i=1}^{n_{in}} w_i x_i
$$

Assume:

- The inputs $x_i$ are independent, each with variance $\text{Var}(x)$.
- The weights $w_i$ are independent, zero-mean, each with variance $\text{Var}(w)$.
- Weights and inputs are independent of each other.

Then, using the identity $\text{Var}(w_ix_i) = \text{Var}(w)\text{Var}(x)$ for independent zero-mean variables:

$$
\text{Var}(z) = \sum_{i=1}^{n_{in}}\text{Var}(w_ix_i) = n_{in}\,\text{Var}(w)\,\text{Var}(x)
$$

So:

$$
\boxed{
\text{Var}(z) = n_{in}\,\text{Var}(w)\,\text{Var}(x)
}
$$

For the output variance to match the input variance, i.e. $\text{Var}(z) = \text{Var}(x)$, we need:

$$
n_{in}\,\text{Var}(w) = 1
\quad\Longrightarrow\quad
\text{Var}(w) = \frac{1}{n_{in}}
$$

This is the basic idea. The forward pass wants $\text{Var}(w) = 1/n_{in}$.

But there's a second constraint: **the backward pass**. During backpropagation, gradients flow through the transpose of the same weight matrix, and by the same argument, the backward pass wants:

$$
\text{Var}(w) = \frac{1}{n_{out}}
$$

Xavier and Kaiming initialization differ mainly in **how they resolve this tension**, and in what activation function they assume downstream.

---

#### 3. Xavier / Glorot Initialization

Xavier initialization balances the forward-pass requirement ($1/n_{in}$) and the backward-pass requirement ($1/n_{out}$) by taking their **harmonic-mean-like compromise**: the average of the two.

$$
\text{Var}(w) = \frac{2}{n_{in}+n_{out}}
$$

Therefore the standard deviation used to sample each weight is:

$$
\boxed{
\sigma_{xavier} = \sqrt{\frac{2}{n_{in}+n_{out}}}
}
$$

and each weight is drawn as:

$$
w \sim \mathcal{N}\left(0,\ \sigma_{xavier}^2\right)
$$

Here:

- $n_{in}$ = `fan_in` = number of input units to the layer
- $n_{out}$ = `fan_out` = number of output units of the layer

This derivation assumes a **linear or symmetric, zero-centered activation** (like `tanh`), since it does not account for an activation function that kills half the signal (like ReLU does).

**Weight matrix shape.** Following the convention used earlier in this document, a weight matrix is shaped `(fan_out, fan_in)` so that:

$$
z = Wx, \qquad W:(fan\_out,\ fan\_in),\ \ x:(fan\_in,)
$$

---

#### 4. Kaiming / He Initialization

ReLU zeroes out roughly **half** of its inputs (everything negative becomes exactly 0). This means the *effective* variance surviving through a ReLU layer is only about half of what it would be with a symmetric activation.

To compensate, Kaiming initialization only balances the **forward pass** ($n_{in}$), but doubles the variance to counteract the fact that ReLU discards half the signal:

$$
\text{Var}(w) = \frac{2}{n_{in}}
$$

So the standard deviation is:

$$
\boxed{
\sigma_{kaiming} = \sqrt{\frac{2}{n_{in}}}
}
$$

and each weight is drawn as:

$$
w \sim \mathcal{N}\left(0,\ \sigma_{kaiming}^2\right)
$$

**Where does the factor of 2 come from, precisely?**

For a zero-mean, symmetric pre-activation $z$, ReLU keeps the positive half and zeros the negative half. It can be shown that:

$$
\text{Var}\big(\text{ReLU}(z)\big) = \frac{1}{2}\text{Var}(z)
$$

So if we want the **output** of the ReLU layer to have the same variance as the **input** to the previous layer, we need the pre-activation variance to be **twice** as large as in the Xavier derivation, which is exactly why the numerator becomes `2` instead of `1`, divided only by `n_in` (the forward-pass-only version):

$$
\text{Var}(z) = n_{in}\,\text{Var}(w) \overset{!}{=} 2 \cdot \text{Var}(x)
\quad\Longrightarrow\quad
\text{Var}(w) = \frac{2}{n_{in}}
$$

This is why Kaiming initialization is the standard choice for networks that use ReLU (or its variants like Leaky ReLU, GELU, etc.), while Xavier remains the standard choice for `tanh`/`sigmoid`-based networks.

---

#### 5. Side-by-Side Comparison

| | Xavier / Glorot | Kaiming / He |
|---|---|---|
| Best for | `tanh`, `sigmoid` (symmetric) | ReLU (and variants) |
| Variance target | $\dfrac{2}{n_{in}+n_{out}}$ | $\dfrac{2}{n_{in}}$ |
| Std used | $\sqrt{\dfrac{2}{n_{in}+n_{out}}}$ | $\sqrt{\dfrac{2}{n_{in}}}$ |
| Accounts for | Balance of forward + backward pass | ReLU zeroing ~half the signal |
| Reference | Glorot & Bengio, 2010 | He et al., 2015 |

---

#### 6. Implementing `xavier_init`

The function needs to:

1. Seed the RNG with `torch.manual_seed(0)` so results are reproducible.
2. Compute $\sigma_{xavier} = \sqrt{2 / (fan\_in + fan\_out)}$.
3. Sample a `(fan_out, fan_in)` matrix from $\mathcal{N}(0,1)$ using `torch.randn`, then scale it by $\sigma_{xavier}$ (scaling a standard normal sample by $\sigma$ produces a sample from $\mathcal{N}(0, \sigma^2)$).
4. Round to 4 decimal places and convert to a nested Python list.

```python
def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
    torch.manual_seed(0)

    denom = fan_in + fan_out
    std = (2.0 / denom) ** 0.5

    return (torch.randn(fan_out, fan_in) * std).round(decimals=4).tolist()
```

Why `torch.randn(fan_out, fan_in) * std` and not `torch.randn(fan_out, fan_in, std=std)`? Because `torch.randn` always samples from the **standard normal** $\mathcal{N}(0,1)$. Multiplying every sample by a scalar $\sigma$ rescales the distribution to $\mathcal{N}(0,\sigma^2)$, since:

$$
X \sim \mathcal{N}(0,1) \quad\Longrightarrow\quad \sigma X \sim \mathcal{N}(0,\sigma^2)
$$

---

#### 7. Implementing `kaiming_init`

Same structure, but with $\sigma_{kaiming} = \sqrt{2/fan\_in}$ — note that `fan_out` is not used at all in the Kaiming formula, since it only cares about preserving variance on the forward pass through a ReLU:

```python
def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
    torch.manual_seed(0)
    std = (2.0 / fan_in) ** 0.5

    return (torch.randn(fan_out, fan_in) * std).round(decimals=4).tolist()
```

---

#### 8. `check_activations`: Watching Variance Propagate Through Depth

This function is a small experiment that demonstrates **why** the initialization formulas above matter in practice. It builds a stack of `num_layers` linear layers (each followed by ReLU), initializes every layer's weights with the chosen scheme, pushes a single random input vector through the whole stack, and records the **standard deviation of the activations after every layer**.

```python
def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
    torch.manual_seed(0)

    weights = []
    for i in range(num_layers):
        fan_in  = input_dim if i == 0 else hidden_dim
        fan_out = hidden_dim

        if init_type == 'xavier':
            std = (2.0 / (fan_in + fan_out)) ** 0.5
        elif init_type == 'kaiming':
            std = (2.0 / fan_in) ** 0.5
        else:
            std = 1.0

        weights.append(torch.randn(fan_out, fan_in) * std)

    x = torch.randn(input_dim)

    stds = []
    for W in weights:
        x = torch.relu(W @ x)
        stds.append(round(float(x.std()), 2))

    return stds
```

**Step-by-step breakdown:**

1. **Building the weight list.** For layer `i = 0`, the input dimension is `input_dim` (the original input size). For every later layer, the input dimension is `hidden_dim` (the output size of the previous layer). Every layer's output size is `hidden_dim`. This mirrors a real MLP:

$$
\text{input\_dim} \rightarrow \text{hidden\_dim} \rightarrow \text{hidden\_dim} \rightarrow \dots \rightarrow \text{hidden\_dim}
$$

2. **Choosing `std` per `init_type`.** This is exactly the formula derived above:
   - `'xavier'` → $\sigma = \sqrt{2/(n_{in}+n_{out})}$
   - `'kaiming'` → $\sigma = \sqrt{2/n_{in}}$
   - anything else (e.g. `'none'` / naive) → $\sigma = 1$, which represents **no scaling at all** — the "bad" baseline that the experiment is meant to contrast against.

3. **Forward pass through the stack.** Starting from a random input $x \sim \mathcal{N}(0,1)^{input\_dim}$, each layer computes:

$$
x \leftarrow \text{ReLU}(Wx)
$$

   and the standard deviation of the resulting vector is recorded.

4. **Interpretation of the output.** The returned list `stds` shows how the "spread" of the activations evolves with depth:
   - With **naive initialization** (`std = 1`), the variance typically **explodes**, so `stds` grows rapidly layer after layer.
   - With **Kaiming initialization**, since it is specifically designed to counteract the variance loss from ReLU, `stds` should stay **roughly constant** across layers.
   - With **Xavier initialization** feeding into ReLU layers (a mismatch, since Xavier assumes a symmetric activation), the variance will typically **shrink** with depth, because Xavier under-corrects for the signal ReLU discards.

This is precisely the practical justification for **matching the initialization scheme to the activation function**: Kaiming for ReLU networks, Xavier for `tanh`/`sigmoid` networks.

---

#### 9. Complete Solution

```python
import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)

        denom = fan_in + fan_out
        std = (2.0 / denom) ** 0.5

        return (torch.randn(fan_out, fan_in) * std).round(decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = (2.0 / fan_in) ** 0.5

        return (torch.randn(fan_out, fan_in) * std).round(decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)

        weights = []
        for i in range(num_layers):
            fan_in  = input_dim if i == 0 else hidden_dim
            fan_out = hidden_dim

            if init_type == 'xavier':
                std = (2.0 / (fan_in + fan_out)) ** 0.5
            elif init_type == 'kaiming':
                std = (2.0 / fan_in) ** 0.5
            else:
                std = 1.0

            weights.append(torch.randn(fan_out, fan_in) * std)

        x = torch.randn(input_dim)

        stds = []
        for W in weights:
            x = torch.relu(W @ x)
            stds.append(round(float(x.std()), 2))

        return stds
```

---

#### 10. Key Takeaways

$$
\boxed{
\sigma_{xavier} = \sqrt{\frac{2}{n_{in}+n_{out}}}
\qquad\qquad
\sigma_{kaiming} = \sqrt{\frac{2}{n_{in}}}
}
$$

- Both schemes exist to keep the **variance of activations (and gradients) stable across many layers**.
- **Xavier** balances the forward and backward pass, and assumes a **symmetric** activation (`tanh`, `sigmoid`).
- **Kaiming** only balances the forward pass but **doubles** the variance to compensate for **ReLU zeroing out half the signal**.
- Weight matrices follow the same `(fan_out, fan_in)` convention used throughout this document, so `z = Wx` matches dimensions correctly.
- `torch.randn(...) * std` is the standard way to sample from $\mathcal{N}(0, \sigma^2)$ in PyTorch, since `torch.randn` itself only ever samples from $\mathcal{N}(0,1)$.
- `check_activations` is a diagnostic tool: it empirically demonstrates that mismatching the initialization scheme to the activation function causes activation statistics to drift (explode or vanish) with network depth, while a matching scheme (Kaiming + ReLU) keeps them stable.