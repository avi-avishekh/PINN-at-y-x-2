# Tiny PINN — Solving an ODE with Physics-Informed Neural Networks

A small beginner-friendly Physics-Informed Neural Network (PINN) project built using PyTorch.

This project demonstrates how a neural network can learn the solution of a differential equation using the governing physics and boundary condition instead of traditional labeled training data.

---

## Problem Statement

We solve the following first-order ordinary differential equation:

du/dx=2x

with the boundary condition:
u(0) = 1

The analytical/exact solution is:
u(x) = x^2 + 1

The objective of the PINN is to learn:

u_theta(x) = x^2 + 1  (approx)

where (u_theta(x)) is the neural network's prediction.

---

## What is a PINN?

A Physics-Informed Neural Network combines:

- Neural Networks
- Differential Equations
- Automatic Differentiation
- Physics-based Loss Functions
- Optimization

Instead of training the network using known input-output labels, the PINN is trained by forcing the network to satisfy the governing differential equation and boundary conditions.

### Basic idea

```text
x
↓
Neural Network
↓
uθ(x)
↓
Automatic Differentiation
↓
duθ/dx
↓
Physics Residual
↓
Loss
↓
Backpropagation
↓
Adam Optimizer
↓
Updated Network
↓
Repeat
