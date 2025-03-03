"""
A PyTorch Tensor is conceptually identical to a numpy array: a Tensor is an n-dimensional array, and PyTorch provides many functions for operating on these Tensors. Behind the scenes, Tensors can keep track of a computational graph and gradients, but they're also useful as a generic tool for scientific computing. Also unlike numpy, PyTorch Tensors can utilize GPUs to accelerate their numeric computations. To run a PyTorch Tensor on GPU, you simply need to specify the correct device. Here we use PyTorch Tensors to fit a third order polynomial to sine function. Like the numpy example above we need to manually implement the forward and backward passes through the network

Reference: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#tensors
"""

# -*- coding: utf-8 -*-
import math
import torch


DTYPE = torch.float
device = torch.device("cpu")

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=DTYPE)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=DTYPE)
b = torch.randn((), device=device, dtype=DTYPE)
c = torch.randn((), device=device, dtype=DTYPE)
d = torch.randn((), device=device, dtype=DTYPE)

LEARNING_RATE = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x**2 + d * x**3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x**2).sum()
    grad_d = (grad_y_pred * x**3).sum()

    # Update weights using gradient descent
    a -= LEARNING_RATE * grad_a
    b -= LEARNING_RATE * grad_b
    c -= LEARNING_RATE * grad_c
    d -= LEARNING_RATE * grad_d


print(f"Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3")
