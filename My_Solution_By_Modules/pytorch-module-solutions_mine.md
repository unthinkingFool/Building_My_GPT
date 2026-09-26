# Build Your GPT — Solutions

## Third Module: PyTorch

### 1. PyTorch Basics

This solution demonstrates four common PyTorch tensor operations:

1. **Reshape** — changes the shape of a tensor without changing its data.
2. **Average** — calculates the column-wise mean across rows.
3. **Concatenate** — joins two tensors side-by-side.
4. **MSE Loss** — calculates the Mean Squared Error between predictions and targets.

---

#### 1. Reshape

The `reshape()` method changes a tensor from shape `(M, N)` into:

$$
\left(\frac{M \times N}{2},\ 2\right)
$$

The total number of elements remains unchanged.

For example:

```text
Original shape: (2, 4)
New shape:      (4, 2)
```

The implementation gets the original dimensions:

```python
M, N = to_reshape.shape
```

Then calculates the new shape:

```python
rows_new = M * N // 2
cols_new = 2

new_shape = (rows_new, cols_new)
```

Finally:

```python
return torch.reshape(to_reshape, new_shape)
```

---

#### 2. Average

To calculate the **column-wise mean**, we use:

```python
torch.mean(to_avg, dim=0)
```

`dim=0` means that PyTorch averages across the rows, leaving one value for each column.

For example:

$$
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{bmatrix}
$$

The column-wise mean is:

$$
\left[\frac{1+3+5}{3},\ \frac{2+4+6}{3}\right] = [3,\ 4]
$$

Therefore:

```python
return torch.mean(to_avg, dim=0)
```

returns the average of each column.

---

#### 3. Concatenate

`torch.cat()` is used to join tensors together.

The code uses:

```python
torch.cat((cat_one, cat_two), dim=1)
```

Since `dim=1` represents the column dimension for a 2D tensor, the tensors are joined **side-by-side**.

For example:

$$
\text{cat\_one} =
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
\qquad
\text{cat\_two} =
\begin{bmatrix}
5 & 6 \\
7 & 8
\end{bmatrix}
$$

The result is:

$$
\begin{bmatrix}
1 & 2 & 5 & 6 \\
3 & 4 & 7 & 8
\end{bmatrix}
$$

The tensors must have compatible dimensions except along the dimension being concatenated.

---

#### 4. Mean Squared Error Loss

Mean Squared Error (MSE) measures the average squared difference between predictions and target values.

PyTorch provides:

```python
torch.nn.functional.mse_loss(prediction, target)
```

Conceptually:

$$
\text{MSE} = \text{mean}\big((\hat{y} - y)^2\big)
$$

A smaller MSE means the predictions are closer to the target values.

---

#### Complete Solution

```python
import torch
import torch.nn
from torchtyping import TensorType


# Round all answers to 4 decimal places: torch.round(tensor, decimals=4)
class Solution:

    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # Reshape (M, N) tensor to (M*N/2, 2)
        # Use torch.reshape(tensor, new_shape)
        M, N = to_reshape.shape

        rows_new = M * N // 2
        cols_new = 2

        new_shape = (rows_new, cols_new)

        return torch.reshape(to_reshape, new_shape)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # Compute column-wise mean (average across rows)
        # Use torch.mean(tensor, dim=0)
        return torch.mean(to_avg, dim=0)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # Join two tensors side-by-side along dim=1
        # Use torch.cat((a, b), dim=1)
        return torch.cat((cat_one, cat_two), dim=1)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # Compute Mean Squared Error between prediction and target
        # Use torch.nn.functional.mse_loss(prediction, target)
        return torch.nn.functional.mse_loss(prediction, target)
```

---

#### Important Points to Remember

| Operation | PyTorch Function | Purpose |
|---|---|---|
| Reshape | `torch.reshape()` | Change tensor shape |
| Mean | `torch.mean(..., dim=0)` | Calculate column-wise average |
| Concatenate | `torch.cat(..., dim=1)` | Join tensors side-by-side |
| MSE Loss | `torch.nn.functional.mse_loss()` | Calculate prediction error |

**Tensor dimensions**, for a 2D tensor:

```text
dim=0 → rows
dim=1 → columns
```

Therefore `torch.mean(x, dim=0)` calculates the mean of each **column**, and `torch.cat((a, b), dim=1)` joins tensors horizontally, side-by-side.

---

### 2. Layer Normalization

This implementation performs the **forward pass of Layer Normalization** for a 1D feature vector using NumPy.

Unlike Batch Normalization (which normalizes across the batch dimension for each feature), Layer Normalization normalizes across the features/channels for each individual sample.

The operation follows three main steps:

1. Calculate the mean and variance across the features of the input vector.
2. Normalize the input using the computed mean and variance.
3. Scale and shift the normalized values using `gamma` and `beta`.

---

#### Problem

Given:

- `x` — a 1D feature vector
- `gamma` — scale parameter with the same length as `x`
- `beta` — shift parameter with the same length as `x`
- `eps = 1e-5` — small value added for numerical stability

The goal is to calculate the layer-normalized output.

The normalization formula is:

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

Then the normalized value is scaled and shifted:

$$
\text{out} = \gamma \hat{x} + \beta
$$

The final result is rounded to **5 decimal places**.

---

#### 1. Calculate the Mean

$$
\mu = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

```python
mean = np.mean(x)
```

The mean represents the average value across the features of the single sample.

---

#### 2. Calculate the Variance

$$
\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2
$$

```python
var = np.mean((x - mean) ** 2)
```

This measures how far the feature values are spread around their mean.

---

#### 3. Normalize the Input

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

```python
x_hat = (x - mean) / ((var + eps) ** 0.5)
```

The purpose of `eps` is to prevent division by zero or numerical instability when the variance is very small.

---

#### 4. Scale and Shift

After normalization, the result is transformed using learnable parameters `gamma` and `beta`:

$$
\text{out} = \gamma \hat{x} + \beta
$$

```python
normalized_arr = gamma * x_hat + beta
```

Here `gamma` controls the **scale** and `beta` controls the **shift**, allowing each layer to adapt the normalized representation dynamically.

---

#### 5. Round the Result

```python
return np.round(normalized_arr, 5)
```

---

#### Complete Code

```python
import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)
        eps = 1e-5

        mean = np.mean(x)
        var = np.mean((x - mean) ** 2)

        x_hat = (x - mean) / ((var + eps) ** 0.5)

        normalized_arr = gamma * x_hat + beta

        return np.round(normalized_arr, 5)
```

---

#### Complete Forward-Pass Flow

```text
Input x
  │
  ├── Calculate mean across features
  │
  ├── Calculate variance across features
  │
  ├── Normalize
  │        (x - mean) / sqrt(var + eps)
  │
  ├── Scale using gamma
  │
  ├── Shift using beta
  │
  ▼
Output
```

#### Important Concepts

| Step | Code | Purpose |
|---|---|---|
| Numerical stability | `eps = 1e-5` | Prevent division by zero |
| Mean | `np.mean(x)` | Calculate feature average |
| Variance | `np.mean((x - mean) ** 2)` | Measure feature spread |
| Normalization | `(x - mean) / sqrt(var + eps)` | Standardize feature values |
| Scale | `gamma * x_hat` | Learnable scaling |
| Shift | `+ beta` | Learnable offset |
| Rounding | `np.round(..., 5)` | Return 5 decimal places |

#### Key Formula

$$
\boxed{
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
\qquad\qquad
\text{out} = \gamma\hat{x} + \beta
}
$$

```text
Input features
  ↓
Center around zero (across features)
  ↓
Normalize variance (across features)
  ↓
Scale with gamma
  ↓
Shift with beta
  ↓
Output
```

This is the basic **Layer Normalization forward pass** for a 1D feature vector.

---

### 3. Batch Normalization (Training & Inference)

This implementation performs the **forward pass of Batch Normalization** for a 2D batch of feature vectors using NumPy, supporting both **training** and **inference** modes.

Batch normalization stabilizes training by normalizing activations across the batch dimension. It also maintains running estimates of mean and variance to be used during inference.

The operation follows distinct paths depending on the mode:

1. **Training Mode** — computes batch mean and variance across the batch dimension, normalizes the input, updates running statistics using momentum, and applies scale/shift.
2. **Inference Mode** — uses pre-computed running mean and variance to normalize the input without computing batch statistics.
3. **Affine Transform** — scales and shifts the normalized values using learnable parameters `gamma` and `beta`.

---

#### Problem

Given:

- `x` — a 2D batch of feature vectors of shape `(batch_size, num_features)`
- `gamma` — scale parameter of shape `(num_features,)`
- `beta` — shift parameter of shape `(num_features,)`
- `running_mean` — running mean estimate across batches
- `running_var` — running variance estimate across batches
- `momentum` — weight factor for updating running statistics
- `eps` — small value added for numerical stability
- `training` — boolean flag indicating training or inference mode

The goal is to calculate the normalized output along with the updated running statistics, all rounded to **4 decimal places** and returned as lists.

---

#### 1. Training Mode Operations

For each feature $i$, compute the mean and variance across the batch dimension:

$$
\mu_i = \frac{1}{m}\sum_{k=1}^{m} x_{k,i}
\qquad
\sigma_i^2 = \frac{1}{m}\sum_{k=1}^{m}(x_{k,i} - \mu_i)^2
$$

```python
mean_i = np.mean(x[:, i])
var_i = np.mean((x[:, i] - mean_i) ** 2)
```

Then, normalize the input batch:

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

```python
x_hat = (x - mean) / ((var + eps) ** 0.5)
```

Finally, update the running statistics using the momentum parameter:

$$
\text{running\_mean} \leftarrow (1-m)\,\text{running\_mean} + m\,\mu
$$

$$
\text{running\_var} \leftarrow (1-m)\,\text{running\_var} + m\,\sigma^2
$$

```python
running_mean = (1 - momentum) * running_mean + momentum * mean
running_var = (1 - momentum) * running_var + momentum * var
```

---

#### 2. Inference Mode Operations

During inference, batch statistics are not computed. Instead, the pre-calculated running statistics are used:

$$
\hat{x} = \frac{x - \text{running\_mean}}{\sqrt{\text{running\_var} + \epsilon}}
$$

```python
x_hat = (x - running_mean) / ((running_var + eps) ** 0.5)
```

---

#### 3. Scale, Shift, and Formatting

After obtaining $\hat{x}$ (from either training or inference), apply the affine transformation:

$$
y = \gamma \hat{x} + \beta
$$

```python
y = gamma * x_hat + beta
```

Finally, all outputs are rounded to 4 decimal places and converted back to Python lists:

```python
return (
    np.round(y, 4).tolist(),
    np.round(running_mean, 4).tolist(),
    np.round(running_var, 4).tolist()
)
```

---

#### Complete Code

```python
import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:

        esp = 1e-5
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)
        beta = np.array(beta, dtype=float)

        running_mean = np.array(running_mean, dtype=float)
        running_var = np.array(running_var, dtype=float)

        if training:
            # x => (batch_size, features)
            num_features = x.shape[1]
            mean = []
            var = []
            for i in range(num_features):
                mean_i = np.mean(x[:, i])
                mean.append(mean_i)

                var_i = np.mean((x[:, i] - mean_i) ** 2)
                var.append(var_i)

            mean = np.array(mean)
            var = np.array(var)

            x_hat = (x - mean) / ((var + eps) ** 0.5)

            # updating the running mean and running var
            m = momentum
            running_mean = (1 - m) * running_mean + m * mean
            running_var = (1 - m) * running_var + m * var

        else:
            x_hat = (x - running_mean) / ((running_var + esp) ** 0.5)

        y = gamma * x_hat + beta

        return (
            np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )
```

---

#### Complete Forward-Pass Flow

```text
Input batch x
  │
  ├── Training Mode?
  │     ├── Yes: Compute batch mean & var, normalize, update running stats
  │     └── No:  Use running mean & var, normalize
  │
  ├── Apply scale (gamma) and shift (beta)
  │
  ▼
Output y & Updated Running Statistics
```

#### Important Concepts

| Step | Code / Concept | Purpose |
|---|---|---|
| Training vs. Inference | `if training:` | Switch behavior between batch stats and running stats |
| Batch Statistics | `np.mean(x[:, i])` | Compute mean/variance per feature across the batch |
| Running Statistics Update | `(1 - m) * running + m * current` | Smoothly track long-term distribution statistics |
| Normalization | `(x - mean) / sqrt(var + eps)` | Standardize feature values |
| Affine Transform | `gamma * x_hat + beta` | Restore representation capacity |
| Rounding & Casting | `np.round(..., 4).tolist()` | Format outputs to 4 decimal places as lists |

#### Key Formula

$$
\boxed{
\begin{aligned}
\mu_B &= \frac{1}{m}\sum_{i=1}^{m} x_i
&
\sigma_B^2 &= \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2 \\[4pt]
\hat{x} &= \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
&
y &= \gamma\hat{x} + \beta \\[4pt]
\text{running\_mean} &\leftarrow (1-\text{momentum})\,\text{running\_mean} + \text{momentum}\,\mu_B
&
\text{running\_var} &\leftarrow (1-\text{momentum})\,\text{running\_var} + \text{momentum}\,\sigma_B^2
\end{aligned}
}
$$

This ensures stable training dynamics and consistent inference performance.

---

### 4. RMS Normalization

This solution implements **Root Mean Square (RMS) Normalization** for a 1D feature vector using NumPy.

RMS Normalization is a simplified, cheaper variant of Layer Normalization: it normalizes the activations using only the **root mean square** of the vector, and it deliberately **omits mean-centering** and the **shift parameter** (`beta`).

---

#### Problem

Given:

- `x` — a 1D feature vector
- `gamma` — scale parameter with the same length as `x`
- `eps` — small value added for numerical stability

The goal is to calculate the RMS-normalized output, scaled by `gamma`, rounded to **4 decimal places**.

---

#### 1. Why RMS Normalization Instead of Layer Normalization?

Layer Normalization re-centers a vector (subtracts the mean) and then rescales it (divides by the standard deviation), before applying a learnable scale **and** shift:

$$
\text{LayerNorm}(x) = \gamma\cdot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

RMS Normalization is built on the observation (from Zhang & Sennrich, *"Root Mean Square Layer Normalization"*, 2019) that the **re-centering step contributes little to the benefits** of Layer Normalization — most of the stabilizing effect comes from **re-scaling**. So RMSNorm drops:

- the **mean subtraction** (no re-centering), and
- the **shift parameter `beta`** (no additive bias).

This makes it both **simpler** and **cheaper to compute** — no mean needs to be calculated at all — which is one reason it is widely used in modern large language models (e.g. LLaMA, T5, Gemma) in place of standard LayerNorm.

---

#### 2. The RMS Statistic

Instead of variance around the mean, RMSNorm computes the **root mean square** of the raw feature values:

$$
\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}
$$

Notice this is exactly the same shape as the standard-deviation formula used in Layer Normalization, **except that `x_i` is used directly instead of `(x_i - mean)`.** In other words, RMS treats the input as if its mean were already zero, and simply measures its magnitude.

```python
rms = (np.mean(x ** 2) + eps) ** 0.5
```

The `eps` term sits **inside** the square root here (rather than being added after, as in LayerNorm) — it plays the same role: preventing a division by (or near) zero when every element of `x` is very small.

---

#### 3. Normalize by the RMS

The input is divided by this RMS statistic:

$$
\hat{x} = \frac{x}{\text{RMS}(x)}
$$

```python
x_hat = x / rms
```

This rescales `x` so that its root-mean-square magnitude becomes approximately 1, without ever shifting it to be zero-centered.

---

#### 4. Scale with Gamma (No Beta)

Unlike Layer Normalization, RMSNorm applies **only a learnable scale**, never a shift:

$$
\text{out} = \gamma \odot \hat{x}
$$

```python
rms_normalized = gamma * x_hat
```

There is no `+ beta` term anywhere in RMS Normalization — this is the defining structural difference from Layer Normalization.

---

#### 5. Round the Result

```python
return np.round(rms_normalized, 4).tolist()
```

---

#### Complete Code

```python
import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)

        rms = (np.mean(x ** 2) + eps) ** 0.5

        x_hat = x / rms

        rms_normalized = gamma * x_hat

        return np.round(rms_normalized, 4).tolist()
```

---

#### Complete Forward-Pass Flow

```text
Input x
  │
  ├── Calculate mean of squares across features
  │        mean(x²)
  │
  ├── Take square root (+ eps) → RMS(x)
  │
  ├── Normalize
  │        x / RMS(x)
  │
  ├── Scale using gamma
  │        (no shift / no beta)
  │
  ▼
Output
```

#### Important Concepts

| Step | Code | Purpose |
|---|---|---|
| No mean subtraction | *(omitted entirely)* | RMSNorm skips re-centering |
| RMS statistic | `(np.mean(x**2) + eps) ** 0.5` | Measures magnitude of `x` directly |
| Numerical stability | `+ eps` inside the square root | Prevent division by zero |
| Normalization | `x / rms` | Rescale `x` to unit RMS magnitude |
| Scale only | `gamma * x_hat` | Learnable scaling — **no `beta`** |
| Rounding & casting | `np.round(..., 4).tolist()` | Format output to 4 decimal places as a list |

#### RMSNorm vs. LayerNorm at a Glance

| | Layer Normalization | RMS Normalization |
|---|---|---|
| Re-centers (subtracts mean)? | Yes | **No** |
| Statistic used | Variance $\sigma^2$ around the mean | Mean of squares (RMS) |
| Learnable shift (`beta`)? | Yes | **No** |
| Learnable scale (`gamma`)? | Yes | Yes |
| Formula | $\gamma\cdot\dfrac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$ | $\gamma\cdot\dfrac{x}{\sqrt{\text{mean}(x^2)+\epsilon}}$ |
| Compute cost | Higher (needs mean + variance) | Lower (needs only mean of squares) |

#### Key Formula

$$
\boxed{
\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}
\qquad\qquad
\text{out} = \gamma\cdot\frac{x}{\text{RMS}(x)}
}
$$

```text
Input features
  ↓
Measure magnitude (root mean square, no centering)
  ↓
Normalize by RMS
  ↓
Scale with gamma
  ↓
Output   (no beta / no shift)
```

This is the RMS Normalization forward pass for a 1D feature vector — a cheaper, scale-only alternative to Layer Normalization.
