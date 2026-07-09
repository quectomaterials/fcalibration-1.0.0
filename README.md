# fcalibration: fuzzy probability calibration
This version does not support parallel computing.

## Usage example
```
import numpy as np
from fcalibration import FcalibrationClass

# Probabilities obtained by a binary classifier with the following label map:
#   Class 0: State 0
#   Class 1: State 1 or State 2
m_train = np.load('m_train.npy')

# Probabilities obtained by a binary classifier with the following label map:
#   Class 0: State 0
#   Class 1: State 2
n_train = np.load('n_probs.npy')

# True labels follow the mapping:
#   Class 0: State 0
#   Class 1: State 1 or State 2
y_train = np.load('y_train.npy')

fcalibration = FcalibrationClass()
fcalibration.fit(m_train, n_train, y_train)

# Probabilities obtained by a binary classifier with the following label map:
#   Class 0: State 0
#   Class 1: State 1 or State 2
m_test = np.load('m_test.npy')

# Probabilities obtained by a binary classifier with the following label map:
#   Class 0: State 0
#   Class 1: State 2
n_test = np.load('n_test.npy')

calibrated_m_test = fcalibration.predict_proba(m_test, n_test)
```

## Initialization arguments

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **self.params** | numpy.ndarray | `np.array([0.0, 0.0, 0.0, 0.0])` | Internal 4-element weight array updated during training:<br>• `self.params[0]`, `self.params[2]` — scale coefficients.<br>• `self.params[1]`, `self.params[3]` — bias coefficients. |

## fit method arguments

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **m_train** | array-like | Required | An array-like object containing probability predictions from a binary classifier, where the target classes are mapped as follows: <br> • Class 0: State 0 <br> • Class 1: State 1 or State 2 |
| **n_train** | array-like | Required |  An array-like object containing probability predictions from a binary classifier, where the target classes are mapped as follows: <br> • Class 0: State 0 <br> • Class 1: State 2 |
| **y_train** | array-like | Required | True labels follow the mapping: <br> • Class 0: State 0 <br> • Class 1: State 1 or State 2. |
| **m_val** | array-like | `None` | For the validation dataset, an array-like object containing probability predictions from a binary classifier, where the target classes are mapped as follows: <br> • Class 0: State 0 <br> • Class 1: State 1 or State 2. |
| **n_val** | array-like | `None` | For the validation dataset, an array-like object containing probability predictions from a binary classifier, where the target classes are mapped as follows: <br> • Class 0: State 0 <br> • Class 1: State 2. |
| **y_val** | array-like | `None` | Target binary labels for validation data. Monitors validation loss if provided. The labels follow the mapping: <br> • Class 0: State 0 <br> • Class 1: State 1 or State 2. |
| **learning_rate** | float | `0.5` | Initial step size for the Adam optimizer. |
| **epochs** | int | `5000` | Maximum number of gradient descent iterations (epochs). |
| **smoothing** | float | `0.0` | Label smoothing coefficient. If `> 0`, dynamically adjusts target labels based on the agreement ("trust") between `m_train` and `n_train`. |
| **patience** | int | `20` | Number of epochs to wait without Log Loss improvement before triggering early stopping. |
| **lr_patience** | int | `5` | Number of epochs to wait without improvement before dropping the learning rate. |
| **factor** | float | `0.5` | Multiplicative reduction factor applied to the learning rate (`current_lr = current_lr * factor`). |
| **min_lr** | float | `1e-5` | Floor limit below which the learning rate cannot be reduced. |
| **l2_reg** | float | `1e-4` | L2 regularization penalty applied to the scale parameters (`self.params[0]` and `self.params[2]`). |
| **verbose** | bool | `True` | If `True`, prints logs every 1000 epochs, as well as optimization alerts (learning rate reduction and early stopping). |

### Returns
* **None**: This method trains the calibration model in-place and updates the internal state (`self.params`). If `self.params[1]` and `self.params[3]` are equal to `0.0` at the start, they are automatically initialized to the log-odds prior based on the `y_train` distribution.

## predict_proba method arguments

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **m_test** | array-like | Required | Probability predictions from the first classifier for the test dataset. |
| **n_test** | array-like | Required | Probability predictions from the second classifier for the test dataset. |

### Returns
* **numpy.ndarray**: A 1D array containing the calibrated predicted probabilities. The resulting values are clipped to the range `[1e-15, 1 - 1e-15]` for numerical stability.

