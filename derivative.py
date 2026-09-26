import torch
import torch.nn as nn


class PINN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),

            nn.Linear(20, 20),
            nn.Tanh(),

            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.network(x)


# 1. Collocation points
x = torch.linspace(
    0,
    1,
    50,
    requires_grad=True
)

x = x.reshape(-1, 1)

# 2. Create model
model = PINN()

# 3. Network prediction
u = model(x)

# 4. First derivative
du_dx = torch.autograd.grad(
    u,
    x,
    grad_outputs=torch.ones_like(u),
    create_graph=True
)[0]

#Physics residual for the PDE: du/dx - 2*x = 0
residual=du_dx-2*x


#Physics Loss
physics_loss=torch.mean(residual**2)

#Boundary Condition Loss:
u_boundary=u[0]
bc_loss=torch.mean((u_boundary-1)**2)

#Total Loss
total_loss=physics_loss+bc_loss


#Optimizer
Optimizer=torch.optim.Adam(model.parameters(),lr=0.001)





print()
print("x shape      :", x.shape)
print("u shape      :", u.shape)
print("du/dx shape  :", du_dx.shape)

print("\nFirst few predictions:")
print(u[:5])

print("\nFirst few derivatives:")
print(du_dx[:5])

print("\nPhysics Residual:", residual[0:5])

print("\nPhysics Loss:", physics_loss)

print("\nBoundary Condition loss:", bc_loss)

print("\nTotal_loss:", total_loss)

print("\nModel Parameter", Optimizer)

#clear old gradients 
Optimizer.zero_grad()

#Backpropagation
total_loss.backward()

print("\nGradient of First layer weight:",model.network[0].weight.grad)


print("Weight before update:")
print(model.network[0].weight)


#Optimizer Update
print("\nOptimizer update: ")
Optimizer.step()

print("weight after update:")
print(model.network[0].weight)

