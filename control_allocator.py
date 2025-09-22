# RBE 577 - Machine Learning for Robotics
# HW 1: Control Allocation via Deep Neural Networks
# ecwenzlaff@wpi.edu

import sys
from typing import Any, Tuple # specified types (esp. those using "Any") primarily convey intention and are not guaranteed to work with static type checking libraries 
import numpy as np
import torch
import vmap_workaround as vw
import matplotlib.pyplot as plt

# Global Variables and Parameters:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
F1_bounds = (-1e4, 1e4)
F2_bounds = (-5e3, 5e3)
F3_bounds = (-5e3, 5e3)
alpha2_bounds = (-180.0, 180.0)
alpha3_bounds = (-180.0, 180.0)
l1, l2, l3, l4 = -14.0, 14.5, -2.7, 2.7

def randomWalkTensor(tensorsize: Tuple[int, int], maxstepvector: torch.FloatTensor) -> torch.FloatTensor:
    # Input/output formatting anticipates state variables in rows and sampled states in columns:
    sample_stepsizes = torch.rand(tensorsize, dtype=torch.float32).to(device)                      # generates floats from 0 to 1.0
    sample_stepsizes[:,0] = 0.0
    sign_tensor = torch.randint(low=-1, high=2, size=tensorsize, dtype=torch.float32).to(device)    # generates ints from [low, high)
    #sign_tensor[torch.where(sign_tensor == 0.0)] = -1.0    # use this when [low, high) is [0, 2) to keep samples in "constant motion"
    scaled_steps = (maxstepvector.to(device) * (sign_tensor * sample_stepsizes)).to(device)    # element-wise multiplication to scale steps forward or backward
    return torch.cumsum(scaled_steps, dim=1)

def clampSamples(tensorin: torch.FloatTensor) -> torch.FloatTensor:
    # TODO: instead of doing a simple clamp, it might be more advantageous to "redraw a step" in the random walk. If you go
    # down this route, it might be better to call this function from the randomWalkTensor function and add additional inputs 
    # to this function to account for random walk parameters/constraints.
    tensorout = torch.zeros((tensorin.shape[0], tensorin.shape[1]), dtype=torch.float32).to(device)
    bound_vector = [F1_bounds, F2_bounds, F3_bounds, alpha2_bounds, alpha3_bounds]
    for i in range(0, len(bound_vector)):
        tensorout[i,:] = torch.clamp(tensorin[i,:], bound_vector[i][0], bound_vector[i][1])
    return tensorout

def forcesToTorques(forcetensor: torch.FloatTensor) -> torch.FloatTensor:    # input dimensions: [5,N], output dimensions: [3,N]
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
    # so I'll need to try using vmap or list comprehension to apply the 5x5 transform to each 5x1 column...
    # torque_tensor = (transform_tensor @ forcetensor)[0,0:3,0:3] 
    def inplacetransform(i: torch.Tensor) -> torch.Tensor:
        return torch.matmul(transform_tensor[i,:,:], forcetensor[:,i])
    torque_tensor = vw.vmap(inplacetransform)(torch.arange(0,forcetensor.shape[1])).to(device)
    #torque_tensor = torch.stack([transform_tensor[i,:,:] @ forcetensor[:,i] for i in range(0,forcetensor.shape[1])]).T[0:3,:].to(device)
    return torque_tensor[:,0:3,0].T

# TODO: need to implement the Encoder/Decoder neural net architecture

# TODO: need to incorporate one of pytorch's data loaders somewhere to split data into (train, validation, test) sets

if __name__ == '__main__':
    print(f"Python Version = {sys.version}, Torch Device = {device}")
    test_tensor = randomWalkTensor((5,int(1e6)), torch.tensor([20, 5, 5, 0.5, 0.5]).reshape(-1,1)).to(device)
    test_tensor = clampSamples(test_tensor)
    torque_tensor = forcesToTorques(test_tensor)
    
    # Plot everything:
    F1_samples = test_tensor.cpu().detach().numpy()[0,:].reshape(-1,1)#[0:300,:]
    F2_samples = test_tensor.cpu().detach().numpy()[1,:].reshape(-1,1)#[0:300,:]
    F3_samples = test_tensor.cpu().detach().numpy()[2,:].reshape(-1,1)#[0:300,:]
    alpha2_samples = test_tensor.cpu().detach().numpy()[3,:].reshape(-1,1)#[0:300,:]
    alpha3_samples = test_tensor.cpu().detach().numpy()[4,:].reshape(-1,1)#[0:300,:]
    tau1_samples = torque_tensor.cpu().detach().numpy()[0,:].reshape(-1,1)#[0:300,:]
    tau2_samples = torque_tensor.cpu().detach().numpy()[1,:].reshape(-1,1)#[0:300,:]
    tau3_samples = torque_tensor.cpu().detach().numpy()[2,:].reshape(-1,1)#[0:300,:]
    fig_F1, ax_F1 = plt.subplots(1,1)
    fig_F2, ax_F2 = plt.subplots(1,1)
    fig_F3, ax_F3 = plt.subplots(1,1)
    fig_alpha2, ax_alpha2 = plt.subplots(1,1)
    fig_alpha3, ax_alpha3 = plt.subplots(1,1)
    fig_tau1, ax_tau1 = plt.subplots(1,1)
    fig_tau2, ax_tau2 = plt.subplots(1,1)
    fig_tau3, ax_tau3 = plt.subplots(1,1)
    ax_F1.plot(np.arange(F1_samples.shape[0]), F1_samples, '.-')
    ax_F1.set_title("F1")
    ax_F2.plot(np.arange(F2_samples.shape[0]), F2_samples, '.-')
    ax_F2.set_title("F2")
    ax_F3.plot(np.arange(F3_samples.shape[0]), F3_samples, '.-')
    ax_F3.set_title("F3")
    ax_alpha2.plot(np.arange(alpha2_samples.shape[0]), alpha2_samples, '.-')
    ax_alpha2.set_title("Alpha2")
    ax_alpha3.plot(np.arange(alpha3_samples.shape[0]), alpha3_samples, '.-')
    ax_alpha3.set_title("Alpha3")
    ax_tau1.plot(np.arange(tau1_samples.shape[0]), tau1_samples, '.-')
    ax_tau1.set_title("Tau1")
    ax_tau2.plot(np.arange(tau2_samples.shape[0]), tau2_samples, '.-')
    ax_tau2.set_title("Tau2")
    ax_tau3.plot(np.arange(tau3_samples.shape[0]), tau3_samples, '.-')
    ax_tau3.set_title("Tau3")
    plt.show()