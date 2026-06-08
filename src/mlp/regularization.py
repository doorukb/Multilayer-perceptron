from __future__ import annotations
import numpy as np

# calculate the l2 penalty for the model
# a Gaussian prior favoring small weights penalizes complexity without shrinking the bias offset
def l2_penalty(my_mlp: dict[str, np.ndarray], lmbda: float) -> float:
    if lmbda == 0.0:
        return 0.0
    sq_sum = sum(np.sum(W[:-1, :] ** 2) for W in my_mlp.values())
    return 0.5 * lmbda * float(sq_sum)