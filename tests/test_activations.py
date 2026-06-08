import numpy as np
import pytest
from mlp.activations import sigmoid_forward, sigmoid_backward, tanh_forward, tanh_backward

def test_sigmoid_forward_known_values():
    assert np.isclose(sigmoid_forward(np.array([0.0])), 0.5)
    assert sigmoid_forward(np.array([100.0]))[0] > 0.99
    assert sigmoid_forward(np.array([-100.0]))[0] < 0.01

def test_sigmoid_backward_matches_finite_difference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    eps = 1e-6
    numerical = (sigmoid_forward(x + eps) - sigmoid_forward(x - eps)) / (2 * eps)
    analytical = sigmoid_backward(x)
    assert np.allclose(numerical, analytical, atol=1e-5)

def test_tanh_forward_known_values():
    assert np.isclose(tanh_forward(np.array([0.0])), 0.0)
    assert tanh_forward(np.array([100.0]))[0] > 0.99
    assert tanh_forward(np.array([-100.0]))[0] < -0.99

def test_tanh_backward_matches_finite_difference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    eps = 1e-6
    numerical = (tanh_forward(x + eps) - tanh_forward(x - eps)) / (2 * eps)
    analytical = tanh_backward(x)
    assert np.allclose(numerical, analytical, atol=1e-5)