# Build Your GPT — Solutions

## Fourth Module: Training a Model

### 1. Training Loop — Batch Gradient Descent for Linear Regression

#### Problem

Implement a full **training loop** that fits a linear regression model using **batch gradient descent**.

The model is:

$$
\hat{y} = Xw + b
$$

where:

- $X$ = input feature matrix, shape $(n, d)$ — $n$ samples, $d$ features
- $w$ = weight vector, shape $(d,)$
- $b$ = scalar bias
- $\hat{y}$ = predicted output, shape $(n,)$

The loss function is **Mean Squared Error (MSE)**:

$$
L = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2
$$

The function must:

1. Initialize $w = \mathbf{0}$ and $b = 0$.
2. Run gradient descent for a fixed number of `epochs`.
3. At every epoch, compute predictions, the loss gradient, and update $w$ and $b$ using the learning rate `lr`.
4. Return the final $(w, b)$, rounded to 5 decimal places.

---

#### 1. The Forward Pass

For the whole batch of $n$ samples at once (vectorized), the prediction is:

$$
\hat{y} = Xw + b
$$

In NumPy, this is a matrix–vector product plus a scalar broadcast across every row:

```python
y_hat = X @ w + b
```

Here `X` has shape `(n, d)` and `w` has shape `(d,)`, so `X @ w` has shape `(n,)` — one prediction per sample — and adding the scalar `b` broadcasts it to every element.

---

#### 2. The Loss

$$
L = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2
$$

```python
loss = np.mean((y_hat - y) ** 2)
```

Although the loop below never actually *uses* `loss` (the gradients are computed directly), it's useful to compute — in practice you'd log it to track convergence.

---

#### 3. Deriving the Gradient with Respect to $w$

This is the core of the exercise. Start from the loss for a single sample $i$:

$$
L_i = (\hat{y}_i - y_i)^2, \qquad \hat{y}_i = \sum_{k=1}^{d} X_{i,k}w_k + b
$$

Differentiate with respect to a single weight $w_k$ using the chain rule:

$$
\frac{\partial L_i}{\partial w_k}
=
\frac{\partial L_i}{\partial \hat{y}_i}
\cdot
\frac{\partial \hat{y}_i}{\partial w_k}
=
2(\hat{y}_i - y_i) \cdot X_{i,k}
$$

Averaging over all $n$ samples (because $L = \frac{1}{n}\sum_i L_i$):

$$
\frac{\partial L}{\partial w_k}
=
\frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)X_{i,k}
$$

Written for the full weight vector, this is exactly a matrix multiplication between the **error vector** $(\hat{y}-y)$ and $X$:

$$
\boxed{
\frac{\partial L}{\partial w} = \frac{2}{n}X^T(\hat{y}-y)
}
$$

Note that $(\hat{y}-y)^T X$ and $X^T(\hat{y}-y)$ produce the same values (just transposed), which is why the code below can write it either way as a 1D dot product.

In NumPy:

```python
dl_dw = (2 * (y_hat - y) @ X) / num_samples
```

Here `(y_hat - y)` has shape `(n,)` and `X` has shape `(n, d)`, so:

$$
(n,) \cdot (n,d) \;\rightarrow\; (d,)
$$

which is exactly $X^T(\hat{y}-y)$ collapsed into a 1D dot product — matching the shape of `w`.

---

#### 4. Deriving the Gradient with Respect to $b$

Since $b$ is added identically to every sample's prediction:

$$
\frac{\partial \hat{y}_i}{\partial b} = 1
$$

so, following the same chain rule as above:

$$
\frac{\partial L}{\partial b}
=
\frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)
$$

$$
\boxed{
\frac{\partial L}{\partial b} = \frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)
}
$$

In NumPy:

```python
dl_db = 2 * np.sum(y_hat - y) / num_samples
```

---

#### 5. The Gradient Descent Update Rule

Once both gradients are known, each parameter takes a small step **against** the direction of the gradient (since the gradient points toward increasing loss):

$$
w \leftarrow w - \eta\,\frac{\partial L}{\partial w}
\qquad\qquad
b \leftarrow b - \eta\,\frac{\partial L}{\partial b}
$$

where $\eta$ (`lr`) is the learning rate — how large a step to take at each iteration.

```python
w = w - lr * dl_dw
b = b - lr * dl_db
```

---

#### 6. Why This Loop Converges

Each epoch performs one full pass:

$$
X, w, b \;\rightarrow\; \hat{y} \;\rightarrow\; L \;\rightarrow\; \left(\frac{\partial L}{\partial w}, \frac{\partial L}{\partial b}\right) \;\rightarrow\; w, b \text{ updated}
$$

Repeating this for `epochs` iterations moves $w$ and $b$ progressively downhill along the loss surface. Because MSE for linear regression is a **convex** function (a smooth bowl shape, with no other local minima), gradient descent with a reasonable learning rate is guaranteed to converge toward the global minimum — the same solution that ordinary least squares would find in closed form.

---

#### 7. Complete Solution

```python
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        num_features = X.shape[1]
        num_samples = X.shape[0]
        b = 0
        w = np.zeros(num_features)

        for i in range(epochs):
            y_hat = X @ w + b
            loss = np.mean((y_hat - y) ** 2)

            dl_dw = (2 * (y_hat - y) @ X) / num_samples
            dl_db = 2 * np.sum(y_hat - y) / num_samples

            w = w - lr * dl_dw
            b = b - lr * dl_db

        return np.round(w, 5), np.round(b, 5)
```

---

#### 8. Complete Training-Loop Flow

```text
Initialize w = 0, b = 0
        │
        ▼
┌───────────────────────────┐
│  Repeat for each epoch:   │
│                           │
│   y_hat = X @ w + b       │
│   loss  = mean((y_hat-y)²)│
│                           │
│   dL/dw = (2/n) Xᵀ(ŷ-y)   │
│   dL/db = (2/n) Σ(ŷ-y)    │
│                           │
│   w ← w - lr * dL/dw      │
│   b ← b - lr * dL/db      │
└───────────────────────────┘
        │
        ▼
Return rounded (w, b)
```

---

#### 9. Important Concepts

| Step | Code | Purpose |
|---|---|---|
| Initialization | `w = np.zeros(num_features)`, `b = 0` | Start from a neutral point |
| Forward pass | `y_hat = X @ w + b` | Vectorized prediction for the whole batch |
| Loss | `np.mean((y_hat - y) ** 2)` | MSE across all samples |
| Weight gradient | `(2 * (y_hat - y) @ X) / num_samples` | $\frac{2}{n}X^T(\hat y - y)$ |
| Bias gradient | `2 * np.sum(y_hat - y) / num_samples` | $\frac{2}{n}\sum(\hat y - y)$ |
| Parameter update | `w -= lr * dl_dw`, `b -= lr * dl_db` | Step downhill on the loss surface |
| Rounding | `np.round(..., 5)` | Return 5 decimal places |

---

#### 10. Key Formulas

$$
\boxed{
\hat{y} = Xw + b
\qquad\qquad
L = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2
}
$$

$$
\boxed{
\frac{\partial L}{\partial w} = \frac{2}{n}X^T(\hat{y}-y)
\qquad\qquad
\frac{\partial L}{\partial b} = \frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)
}
$$

$$
\boxed{
w \leftarrow w - \eta\frac{\partial L}{\partial w}
\qquad\qquad
b \leftarrow b - \eta\frac{\partial L}{\partial b}
}
$$

This is **batch gradient descent** — every epoch uses the *entire* dataset `X` to compute one gradient step, as opposed to stochastic gradient descent (one sample at a time) or mini-batch gradient descent (a small subset at a time).