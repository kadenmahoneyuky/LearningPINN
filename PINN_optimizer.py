import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))  # Name of your GPU

def exact_solution(d, w0, t):
    "Defines the analytical solution to the under-damped harmonic oscillator problem above."
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*t)
    exp = torch.exp(-d*t)
    u = exp*2*A*cos
    return u

class FCN(nn.Module):
    "Defines a standard fully-connected network in PyTorch"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
    
loss_time = torch.zeros((0,2))
percent = 0
toc = time.time()

for i in range(0,1001,50):
    for j in range(0,25001,1000):
        torch.cuda.synchronize()
        tic = time.time()
        # first, create some noisy observational data
        torch.manual_seed(123)
        d, w0 = 2, 20
        t_obs = torch.rand(40).view(-1,1).to(device)
        u_obs = exact_solution(d, w0, t_obs).to(device) + 0.04*torch.randn_like(t_obs).to(device)

        torch.manual_seed(123)

        # define a neural network to train
        pinn = FCN(1,1,32,3).to(device)

        # define training points over the entire domain, for the physics loss
        t_physics = torch.linspace(0,1,i).view(-1,1).requires_grad_(True).to(device)

        # train the PINN
        d, w0 = 2, 20
        _, k = 2*d, w0**2

        # treat mu as a learnable parameter
        # TODO: write code here
        mu = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
        losses = []


        # add mu to the optimiser
        # TODO: write code here
        optimiser = torch.optim.Adam(list(pinn.parameters())+[mu],lr=1e-3)

        for i in range(j + 1):
            optimiser.zero_grad()
            
            # compute each term of the PINN loss function above
            # using the following hyperparameters:
            lambda1 = 1e4
            
            # compute physics loss
            u = pinn(t_physics)
            dudt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
            d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]
            loss1 = torch.mean((d2udt2 + mu*dudt + k*u)**2)
            
            # compute data loss
            # TODO: write code here
            u = pinn(t_obs)
            loss2 = torch.mean((u - u_obs)**2)

                
            # backpropagate joint loss, take optimiser step
            loss = loss1 + lambda1*loss2
            loss.backward()
            optimiser.step()
            
            new_row = torch.tensor([[time.time()-tic, loss.item()]])
            loss_time = torch.cat((loss_time, new_row), dim=0)

        if (j % 5000) == 0:
            percent += 1
            print(f"{percent:.0f}% -- {time.time() - toc:.2f} seconds")
            torch.cuda.synchronize()
            toc = time.time()


x = loss_time[:, 0]
y = loss_time[:, 1]

# Create scatter plot
plt.scatter(x, y)
plt.xlabel("Loss")
plt.ylabel("Time")
plt.title("Loss vs. Time")
plt.grid(True)
plt.show()