# RBE 577 - Machine Learning for Robotics
# HW 1: Control Allocation via Deep Neural Networks
# ecwenzlaff@wpi.edu

import sys
from typing import Any  # specified types (esp. those using "Any") primarily convey intention and are not guaranteed to work with static type checking libraries 
import numpy as np
import torch
import matplotlib.pyplot as plt

# Global Variables and Parameters:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
F1_bounds = (-1e4, 1e4)
F2_bounds = (-5e3, 5e3)
F3_bounds = (-5e3, 5e3)
alpha2_bounds = (-180.0, 180.0)
alpha3_bounds = (-180.0, 180.0)
l1, l2, l3, l4 = -14.0, 14.5, -2.7, 2.7

def forces2torques(forcetensor: torch.FloatTensor) -> torch.FloatTensor:    # input dimensions: [5,N], output dimensions: [3,N]
    transform_tensor = torch.zeros((forcetensor.shape[1],5,5), dtype=torch.float32).to(device)
    #torque_tensor = torch.zeros((5,forcetensor.shape[1]), dtype=torch.float32).to(device)
    # Each 5x5xi slice should align with the torque transform in section 3.1. So when the 3x3
    # matrix given by equation (6) is expanded to apply to all 5 input commands, the individual
    # transform for a single set of five input commands becomes:
    #   Transform = [
    #       [ 0,                    cos(alpha2),                      cos(alpha3), 0, 0],
    #       [ 1,                    sin(alpha2),                      sin(alpha3), 0, 0],
    #       [l2,  l1*sin(alpha2)-l3*cos(alpha2),    l1*sin(alpha3)-l4*cos(alpha3), 0, 0],
    #       [ 0,                    0,                                          0, 0, 0],
    #       [0,                     0,                                          0, 0, 0]
    #   ]
    transform_tensor[:,0,1] = torch.cos((np.pi/180)*forcetensor[3,:].reshape(-1))
    transform_tensor[:,0,2] = torch.cos((np.pi/180)*forcetensor[4,:].reshape(-1))
    transform_tensor[:,1,0] = 1.0
    transform_tensor[:,1,1] = torch.sin((np.pi/180)*forcetensor[3,:].reshape(-1))
    transform_tensor[:,1,2] = torch.sin((np.pi/180)*forcetensor[4,:].reshape(-1))
    transform_tensor[:,2,0] = l2
    transform_tensor[:,2,1] = l1*torch.sin((np.pi/180)*forcetensor[3,:].reshape(-1)) - l3*torch.cos((np.pi/180)*forcetensor[3,:].reshape(-1))
    transform_tensor[:,2,2] = l1*torch.sin((np.pi/180)*forcetensor[4,:].reshape(-1)) - l4*torch.cos((np.pi/180)*forcetensor[4,:].reshape(-1))
    # Even though I only care about a single slice, broadcasting here could blow up my GPU memory,
    # so I'll unfortunately need to use list comprehension instead...
    # torque_tensor = (transform_tensor @ forcetensor)[0,0:3,0:3] 
    #def inplacetransform(i):
    #    torque_tensor[:,i] = torch.matmul(transform_tensor[i,:,:], forcetensor[:,i])
    #[torque_tensor[i,:] = (transform_tensor[i,:,:] @ forcetensor[:,i]) for i in range(0,forcetensor.shape[1])]
    #torch.vmap(inplacetransform)(torch.arange(0,forcetensor.shape[1]))
    torque_tensor = torch.stack([transform_tensor[i,:,:] @ forcetensor[:,i] for i in range(0,forcetensor.shape[1])]).T[0:3,:].to(device)
    return torque_tensor


if __name__ == '__main__':
    print(f"Python Version = {sys.version}, Torch Device = {device}")
    sampled_tensor = torch.rand((5,int(1e6)), dtype=torch.float32).to(device)   # should generate numbers from -1.0 to 1.0
    scale_tensor = torch.tensor([
        F1_bounds[1], F2_bounds[1], F3_bounds[1], alpha2_bounds[1], alpha3_bounds[1]
        ], dtype=torch.float32).reshape(-1,1).to(device)
    sign_tensor = torch.randint(low=0, high=2, size=(5,int(1e6)), dtype=torch.float32).to(device)
    sign_tensor[torch.where(sign_tensor == 0.0)] = -1.0
    test_tensor = torch.zeros((5,int(1e6)), dtype=torch.float32).to(device) 
    # Desired element-wise operation sequence: (1) each scale_tensor element multiplied by the associated row in sampled_tensor, 
    # followed by (2) each element in sign_tensor multiplied by each element in sampled_tensor
    test_tensor = sign_tensor * (scale_tensor * sampled_tensor) 
    #print(f"{test_tensor = }, {test_tensor.shape = }")
    torque_tensor = forces2torques(test_tensor)
    
    # Plot everything:
    F1_samples = test_tensor.cpu().detach().numpy()[0,:].reshape(-1,1)
    F2_samples = test_tensor.cpu().detach().numpy()[1,:].reshape(-1,1)
    F3_samples = test_tensor.cpu().detach().numpy()[2,:].reshape(-1,1)
    alpha2_samples = test_tensor.cpu().detach().numpy()[3,:].reshape(-1,1)
    alpha3_samples = test_tensor.cpu().detach().numpy()[4,:].reshape(-1,1)
    tau1_samples = torque_tensor.cpu().detach().numpy()[0,:].reshape(-1,1)
    tau2_samples = torque_tensor.cpu().detach().numpy()[1,:].reshape(-1,1)
    tau3_samples = torque_tensor.cpu().detach().numpy()[2,:].reshape(-1,1)
    fig_F1, ax_F1 = plt.subplots(1,1)
    fig_F2, ax_F2 = plt.subplots(1,1)
    fig_F3, ax_F3 = plt.subplots(1,1)
    fig_alpha2, ax_alpha2 = plt.subplots(1,1)
    fig_alpha3, ax_alpha3 = plt.subplots(1,1)
    fig_tau1, ax_tau1 = plt.subplots(1,1)
    fig_tau2, ax_tau2 = plt.subplots(1,1)
    fig_tau3, ax_tau3 = plt.subplots(1,1)
    ax_F1.plot(np.arange(F1_samples.shape[0]), F1_samples, '.')
    ax_F1.set_title("F1")
    ax_F2.plot(np.arange(F2_samples.shape[0]), F2_samples, '.')
    ax_F2.set_title("F2")
    ax_F3.plot(np.arange(F3_samples.shape[0]), F3_samples, '.')
    ax_F3.set_title("F3")
    ax_alpha2.plot(np.arange(alpha2_samples.shape[0]), alpha2_samples, '.')
    ax_alpha2.set_title("Alpha2")
    ax_alpha3.plot(np.arange(alpha3_samples.shape[0]), alpha3_samples, '.')
    ax_alpha3.set_title("Alpha3")
    ax_tau1.plot(np.arange(tau1_samples.shape[0]), tau1_samples, '.')
    ax_tau1.set_title("Tau1")
    ax_tau2.plot(np.arange(tau2_samples.shape[0]), tau2_samples, '.')
    ax_tau2.set_title("Tau2")
    ax_tau3.plot(np.arange(tau3_samples.shape[0]), tau3_samples, '.')
    ax_tau3.set_title("Tau3")
    plt.show()