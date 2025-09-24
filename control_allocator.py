# RBE 577 - Machine Learning for Robotics
# HW 1: Control Allocation via Deep Neural Networks
# ecwenzlaff@wpi.edu

import sys
from typing import Any, Tuple # specified types (esp. those using "Any") primarily convey intention and are not guaranteed to work with static type checking libraries 
import numpy as np
import torch
import vmap_workaround as vw
import matplotlib.pyplot as plt
import random
import plot_utilities as util

# Global Variables and Parameters:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
F1_bounds = (-1e4, 1e4)
F2_bounds = (-5e3, 5e3)
F3_bounds = (-5e3, 5e3)
alpha2_bounds = (-180.0, 180.0)
alpha3_bounds = (-180.0, 180.0)
l1, l2, l3, l4 = -14.0, 14.5, -2.7, 2.7

def randomWalkTensor(tensorsize: Tuple[int, int], maxstepvector: torch.FloatTensor, initstate: torch.Tensor = None) -> torch.FloatTensor:
    # Input/output formatting anticipates state variables in rows and sampled states in columns:
    sample_stepsizes = torch.rand(tensorsize, dtype=torch.float32).to(device)                      # generates floats from 0 to 1.0
    if (initstate == None):
        sample_stepsizes[:,0] = 0.0
    else:
        sample_stepsizes[:,0] = initstate
    sign_tensor = torch.randint(low=-1, high=2, size=tensorsize, dtype=torch.float32).to(device)    # generates ints from [low, high)
    #sign_tensor[torch.where(sign_tensor == 0.0)] = -1.0    # use this when [low, high) is [0, 2) to keep samples in "constant motion"
    scaled_steps = (maxstepvector.to(device) * (sign_tensor * sample_stepsizes)).to(device)    # element-wise multiplication to scale steps forward or backward
    displacement_tensor = torch.cumsum(scaled_steps, dim=1)
    clamped_tensor = clampSamples(scaled_steps, displacement_tensor, maxstepvector)
    return clamped_tensor

def clampSamples(stepwalktensor: torch.FloatTensor, distancetensor: torch.FloatTensor, stepsizevector: torch.FloatTensor) -> torch.FloatTensor:
    # This function performs "course correction" on the cumulative displacement tensor by redrawing steps that cause the 
    # state variable displacements to go out of their respective bounds: 
    clamped_distance = distancetensor.detach().clone().to(device)
    clamped_walksteps = stepwalktensor.detach().clone().to(device) #torch.zeros((tensorin.shape[0], tensorin.shape[1]), dtype=torch.float32).to(device)
    bound_vector = [F1_bounds, F2_bounds, F3_bounds, alpha2_bounds, alpha3_bounds]
    # Correction factors are defined to prevent overcorrecting the displacement drift. Assuming that the first step that drove 
    # displacement drift out of bounds is "fixed", it may be fine for the steps that follow to go in any direction. So a separate
    # correction factor will be applied for steps after the initial out of bounds step to serve as a "careful" walk near the boundary:
    init_correction_factor = 1.0
    careful_correction_factor = 0.5
    for i in range(0, len(bound_vector)):
        under_idx = torch.where(clamped_distance[i,:].reshape(1,-1) < bound_vector[i][0])
        over_idx = torch.where(clamped_distance[i,:].reshape(1,-1) > bound_vector[i][1])
        while ( (under_idx[1].shape[0] > 0) or (over_idx[1].shape[0] > 0) ):
            if (under_idx[1].shape[0] > 0):
                clamped_walksteps[i,under_idx[1][0]] = stepsizevector.reshape(-1)[i] * init_correction_factor * torch.rand((1,), dtype=torch.float32)
                if (under_idx[1].shape[0] > 1):
                    step_direction = careful_correction_factor * torch.randint(low=-1, high=2, size=(under_idx[0].shape[0]-1,), dtype=torch.float32).to(device)
                    clamped_walksteps[i,under_idx[1][1:]] = stepsizevector.reshape(-1)[i] * step_direction * torch.rand((under_idx[0].shape[0]-1,), dtype=torch.float32).to(device)
            if (over_idx[1].shape[0] > 0):
                clamped_walksteps[i,over_idx[1][0]] = stepsizevector.reshape(-1)[i] * (-init_correction_factor) * torch.rand((1,), dtype=torch.float32)
                if (over_idx[1].shape[0] > 1):
                    step_direction = careful_correction_factor * torch.randint(low=-1, high=2, size=(over_idx[0].shape[0]-1,), dtype=torch.float32).to(device)
                    clamped_walksteps[i,over_idx[1][1:]] = stepsizevector.reshape(-1)[i] * step_direction * torch.rand((over_idx[0].shape[0]-1,), dtype=torch.float32).to(device)
            clamped_distance = torch.cumsum(clamped_walksteps, dim=1)
            under_idx = torch.where(clamped_distance[i,:].reshape(1,-1) < bound_vector[i][0])
            over_idx = torch.where(clamped_distance[i,:].reshape(1,-1) > bound_vector[i][1])
    return clamped_distance

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

if __name__ == '__main__':
    print(f"Python Version = {sys.version}, Torch Device = {device}")
    # Generate datasets:
    test_tensor = randomWalkTensor(
        (5,int(1e6)), 
        torch.tensor([100.0, 10.0, 10.0, 2.0, 2.0]).reshape(-1,1),
        torch.tensor([
            random.uniform(F1_bounds[0], F1_bounds[1]),
            random.uniform(F2_bounds[0], F2_bounds[1]),
            random.uniform(F3_bounds[0], F3_bounds[1]),
            random.uniform(alpha2_bounds[0], alpha2_bounds[1]),
            random.uniform(alpha3_bounds[0], alpha3_bounds[1])
        ]).reshape(-1)).to(device)
    torque_tensor = forcesToTorques(test_tensor)
    T_train, T_validation, T_test = np.split(torque_tensor.cpu().detach().numpy(), 
                                                [
                                                    int(0.7*torque_tensor.shape[1]),    # first split: train = 0->70%
                                                    int(0.8*torque_tensor.shape[1]),    # second split: validation = 70->80%, test = 80->100%
                                                ], axis=1)
    # TODO: might be beneficial to port this splitting code into a function:
    T_train = torch.tensor(T_train, dtype=torch.float32).to(device)
    T_validation = torch.tensor(T_validation, dtype=torch.float32).to(device)
    T_test = torch.tensor(T_test, dtype=torch.float32).to(device)
    T_train = T_train[:, torch.randperm(T_train.shape[1])]  # only shuffle the training data

    # Create LineStructures for plotting:
    F1_samples = util.LineStructure(x=np.arange(0,test_tensor.shape[1]), y=test_tensor.cpu().detach().numpy()[0,:].reshape(-1,1), marker='.') 
    F2_samples = util.LineStructure(x=np.arange(0,test_tensor.shape[1]), y=test_tensor.cpu().detach().numpy()[1,:].reshape(-1,1), marker='.')
    F3_samples = util.LineStructure(x=np.arange(0,test_tensor.shape[1]), y=test_tensor.cpu().detach().numpy()[2,:].reshape(-1,1), marker='.')
    alpha2_samples = util.LineStructure(x=np.arange(0,test_tensor.shape[1]), y=test_tensor.cpu().detach().numpy()[3,:].reshape(-1,1), marker='.')
    alpha3_samples = util.LineStructure(x=np.arange(0,test_tensor.shape[1]), y=test_tensor.cpu().detach().numpy()[4,:].reshape(-1,1), marker='.')
    tau1_samples = util.LineStructure(x=np.arange(0,torque_tensor.shape[1]), y=torque_tensor.cpu().detach().numpy()[0,:].reshape(-1,1))
    tau2_samples = util.LineStructure(x=np.arange(0,torque_tensor.shape[1]), y=torque_tensor.cpu().detach().numpy()[1,:].reshape(-1,1))
    tau3_samples = util.LineStructure(x=np.arange(0,torque_tensor.shape[1]), y=torque_tensor.cpu().detach().numpy()[2,:].reshape(-1,1))
    train_samples_tau1 = util.LineStructure(x=np.arange(0,T_train.shape[1]), y=T_train.cpu().detach().numpy()[0,:], label="Tau1")
    train_samples_tau2 = util.LineStructure(x=np.arange(0,T_train.shape[1]), y=T_train.cpu().detach().numpy()[1,:], label="Tau2")
    train_samples_tau3 = util.LineStructure(x=np.arange(0,T_train.shape[1]), y=T_train.cpu().detach().numpy()[2,:], label="Tau3")
    validate_samples_tau1 = util.LineStructure(x=np.arange(0,T_validation.shape[1]), y=T_validation.cpu().detach().numpy()[0,:], label="Tau1")
    validate_samples_tau2 = util.LineStructure(x=np.arange(0,T_validation.shape[1]), y=T_validation.cpu().detach().numpy()[1,:], label="Tau2")
    validate_samples_tau3 = util.LineStructure(x=np.arange(0,T_validation.shape[1]), y=T_validation.cpu().detach().numpy()[2,:], label="Tau3")
    test_samples_tau1 = util.LineStructure(x=np.arange(0,T_test.shape[1]), y=T_test.cpu().detach().numpy()[0,:], label="Tau1")
    test_samples_tau2 = util.LineStructure(x=np.arange(0,T_test.shape[1]), y=T_test.cpu().detach().numpy()[1,:], label="Tau2")
    test_samples_tau3 = util.LineStructure(x=np.arange(0,T_test.shape[1]), y=T_test.cpu().detach().numpy()[2,:], label="Tau3")

    # Generate plots:
    fig_F1, ax_F1 = util.plotLineStructures([F1_samples], supertitle="F1")[0:2]
    fig_F2, ax_F2 = util.plotLineStructures([F2_samples], supertitle="F2")[0:2]
    fig_F3, ax_F3 = util.plotLineStructures([F3_samples], supertitle="F3")[0:2]
    fig_a2, ax_a2 = util.plotLineStructures([alpha2_samples], supertitle="alpha2")[0:2]
    fig_a3, ax_a3 = util.plotLineStructures([alpha3_samples], supertitle="alpha3")[0:2]
    fig_tau1, ax_tau1 = util.plotLineStructures([tau1_samples], supertitle="Tau1")[0:2]
    fig_tau2, ax_tau2 = util.plotLineStructures([tau2_samples], supertitle="Tau2")[0:2]
    fig_tau3, ax_tau3 = util.plotLineStructures([tau3_samples], supertitle="Tau3")[0:2]
    fig_train, ax_train = util.plotLineStructures([train_samples_tau1, train_samples_tau2, train_samples_tau3], supertitle="Train", splitview=(3,1))[0:2]
    fig_val, ax_val = util.plotLineStructures([validate_samples_tau1, validate_samples_tau2, validate_samples_tau3], supertitle="Validation")[0:2]
    fig_test, ax_test = util.plotLineStructures([test_samples_tau1, test_samples_tau2, test_samples_tau3], supertitle="Test")[0:2]

    plt.show()
