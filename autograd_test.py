import torch

# Step 1: input
x = torch.tensor(2.0, requires_grad=True)

# Step 2: mathematical function
y = x**3

# Step 3: derivative
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]

d2y_dx2=torch.autograd.grad(dy_dx,x)

print("x =", x)
print("y =", y)
print("dy/dx =", dy_dx)
print("d2y_dx2=", d2y_dx2[0])