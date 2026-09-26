import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42);
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
    

#Collocation Points

x=torch.linspace(0,10,1000, requires_grad=True)
x=x.reshape(-1,1)

#Model
model=PINN()

#Optimizer
optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
        
##Loop for Epoch

for epoch in range(10000):
    #Clear Old Gradients
    optimizer.zero_grad()
    
    #Forward Pass
    u=model(x)
    
    #Calculate derivative
    du_dx=torch.autograd.grad(u,x, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    #Physics residual
    residual=du_dx-2*x
    
    # Physics Loss
    physics_loss=torch.mean(residual**2)
    
    #Boundary Condition
    
    u_boundary=u[0]
    bc_loss=(u_boundary-1)**2
    
    # Total Loss
    total_loss=physics_loss+bc_loss
    
    #Backpropagation
    total_loss.backward()
    
    #Update parameters
    optimizer.step()
    
    best_loss=float("inf")
    
    #Print Progress
    if epoch %1000==0:
        print("Epoch: ",epoch,
              "Loss: ", total_loss.item()
              )
    elif total_loss.item()<best_loss:
            best_loss=total_loss.item()
            
print("Best Loss:", best_loss)


#Prediction after traning
u_pred=model(x)

print("\nPredictions: ")
print(u_pred)

#Exact Solution
u_exact= x**2+1

#absolute Error
absolute_error= torch.abs(u_pred-u_exact)

print("\nAbsolute Error:")
print(absolute_error)

mae=torch.mean(absolute_error)
print("\nMAE: ", mae.item())


#Visualization 
x_plot=x.detach().numpy()

u_pred_plot= u_pred.detach().numpy()
u_exact_plot=u_exact.detach().numpy()

plt.figure(figsize=(10,5))

#PINN Prediction
plt.plot(
    x_plot,
    u_pred_plot,
    marker='o',
    linestyle='--',
    label="PINN Prediction"
)

#Exact Solution
plt.plot(
    x_plot,
    u_exact_plot,
    linestyle='-',
    label="Exact Solution"
)

plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("PINN vs Exact Solution")

plt.legend()
plt.grid(True)
plt.show()

