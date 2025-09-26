# RBE 577 - Machine Learning for Robotics
# HW 1: Control Allocation via Deep Neural Networks
# ecwenzlaff@wpi.edu

import sys
from typing import Any, Tuple, Union # specified types (esp. those using "Any") primarily convey intention and are not guaranteed to work with static type checking libraries 
import numpy as np
import torch
import torch.nn as nn
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

def splitTrainValTest(dataset: np.ndarray, 
             slicepoints: Union[Tuple[float] | Tuple[float, float]], 
             shuffleset: Union[Tuple[bool, bool] | Tuple[bool, bool, bool]],
             sliceaxis: int = 1) -> Union[Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    slicelist = []
    for val in slicepoints:
        slicelist.append(int(val*dataset.shape[sliceaxis]))
    if (len(slicepoints)==2) and (len(shuffleset)==3):
        trainset, validationset, testset = np.split(dataset, slicelist, axis=sliceaxis)
        trainset = torch.tensor(trainset, dtype=torch.float32).to(device)
        validationset = torch.tensor(validationset, dtype=torch.float32).to(device)
        testset = torch.tensor(testset, dtype=torch.float32).to(device)
        if (shuffleset[0]):
            trainset = trainset[:, torch.randperm(trainset.shape[sliceaxis])]
        if (shuffleset[1]):
            validationset = validationset[:, torch.randperm(validationset.shape[sliceaxis])]
        if (shuffleset[2]):
            testset = testset[:, torch.randperm(testset.shape[sliceaxis])]
        return trainset, validationset, testset
    elif ((len(slicepoints)==1) and (len(shuffleset)==2)):
        trainset, testset = np.split(dataset, slicelist, axis=sliceaxis)
        trainset = torch.tensor(trainset, dtype=torch.float32).to(device)
        testset = torch.tensor(testset, dtype=torch.float32).to(device)
        if (shuffleset[0]):
            trainset = trainset[:, torch.randperm(trainset.shape[sliceaxis])]
        if (shuffleset[1]):
            testset = testset[:, torch.randperm(testset.shape[sliceaxis])]
        return trainset, testset
    else:
        raise(f"splitTVT error - invalid combination of tuple sizes: {len(slicepoints)=}, {len(shuffleset)=}")

# TODO: turns out we don't need to implement an autoencoder with LSTMs like the paper described, so you'll need to implement 
# an autoencoder purely made up of Fully Connected Neural Networks instead (AutoEncoder_FCN) 
class AutoEncoder_LSTM(nn.Module):
    def __init__(self, lstm_nodes: int = 64, lstm_layers: int = 2, net_dropout: float = 0.0):
        super(AutoEncoder_LSTM, self).__init__()
        self.hidden_size = int(lstm_nodes/lstm_layers)
        self.lstm_layers = lstm_layers
        self.encoder_lstm = nn.LSTM(input_size=3, hidden_size=self.hidden_size, num_layers=lstm_layers, dropout=net_dropout).to(device)
        self.encoder_linear = nn.Sequential(
            nn.Linear(self.hidden_size, 5), 
            nn.Dropout(net_dropout)
        ).to(device)
        self.decoder_lstm = nn.LSTM(input_size=5, hidden_size=self.hidden_size, num_layers=lstm_layers, dropout=net_dropout).to(device)
        self.decoder_linear = nn.Sequential(
            nn.Linear(self.hidden_size, 3),
            nn.Dropout(net_dropout)
        ).to(device)
    def encode(self, generaltorques: torch.FloatTensor):
        # NOTE: generaltorques dimension should be (N-by-D) here, where:
        #   N = number of samples (number of batches)
        #   D = torque vector dimension (number of state variables)
        # 
        # As part of the forward processing, the 2D input tensor will be expanded
        # to 3D in order to pass it into the LSTM; since conceptually, the (N-by-D) tensor
        # could be interpretted as N (1-by-D) tensors.
        gt3d = generaltorques.reshape(1, generaltorques.shape[0], generaltorques.shape[1])
        init_hidden_state = torch.zeros((self.lstm_layers, gt3d.shape[1], self.hidden_size), dtype=torch.float32).to(device)
        init_cell_state = torch.zeros((self.lstm_layers, gt3d.shape[1], self.hidden_size), dtype=torch.float32).to(device)
        lstm_output, lstm_states = self.encoder_lstm(gt3d, (init_hidden_state, init_cell_state))
        linear_out = self.encoder_linear(lstm_output[-1,:,:])
        return linear_out
    def decode(self, generalforces: torch.FloatTensor):
        # NOTE: generalforces dimension should be (N-by-D) here, where:
        #   N = number of samples (number of batches)
        #   D = force vector dimension (number of state variables)
        # 
        # As part of the forward processing, the 2D input tensor will be expanded
        # to 3D in order to pass it into the LSTM; since conceptually, the (N-by-D) tensor
        # could be interpretted as N (1-by-D) tensors.
        gf3d = generalforces.reshape(1, generalforces.shape[0], generalforces.shape[1])
        init_hidden_state = torch.zeros((self.lstm_layers, gf3d.shape[1], self.hidden_size), dtype=torch.float32).to(device)
        init_cell_state = torch.zeros((self.lstm_layers, gf3d.shape[1], self.hidden_size), dtype=torch.float32).to(device)
        lstm_output, lstm_states = self.decoder_lstm(gf3d, (init_hidden_state, init_cell_state))  
        linear_out = self.decoder_linear(lstm_output[-1,:,:])
        return linear_out
    def forward(self, generaltorques: torch.FloatTensor) -> torch.FloatTensor:
        # NOTE: generaltorques dimension should be (N-by-D) here, where:
        #   N = number of samples (number of batches)
        #   D = torque vector dimension (number of state variables)
        # 
        # As part of each component's respective forward processing, the 2D input tensor 
        # will be expanded to 3D in order to pass it into their respective LSTMs.
        allocatedforces = self.encode(generaltorques)
        decodedtorques = self.decode(allocatedforces)
        return decodedtorques

# TODO: Need to incorporate the loss functions described in the paper and then stitch them into a pytorch optimizer
def compute_L1():
    pass
    
def compute_L2():
    pass

def compute_L3():
    pass
    
def compute_L4():
    pass
    
def compute_L5():
    pass

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
    T_train, T_validation, T_test = splitTrainValTest(torque_tensor.cpu().detach().numpy(), (0.7, 0.8), (True, False, False), 1)
    
    # Configure the NN model:
    model = AutoEncoder_LSTM()
    print(f"{model.encode(T_train[:,0:10].T) = }")
    print(f"{model.decode(test_tensor[:,0:10].T) = }")
    print(f"{model(T_train[:,0:10].T) = }")

    """
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
    fig_F1, ax_F1, _ = util.plotLineStructures([F1_samples], supertitle="F1")
    fig_F2, ax_F2, _ = util.plotLineStructures([F2_samples], supertitle="F2")
    fig_F3, ax_F3, _ = util.plotLineStructures([F3_samples], supertitle="F3")
    fig_a2, ax_a2, _ = util.plotLineStructures([alpha2_samples], supertitle="alpha2")
    fig_a3, ax_a3, _ = util.plotLineStructures([alpha3_samples], supertitle="alpha3")
    fig_tau1, ax_tau1, _ = util.plotLineStructures([tau1_samples], supertitle="Tau1")
    fig_tau2, ax_tau2, _ = util.plotLineStructures([tau2_samples], supertitle="Tau2")
    fig_tau3, ax_tau3, _ = util.plotLineStructures([tau3_samples], supertitle="Tau3")
    fig_train, ax_train, _ = util.plotLineStructures([train_samples_tau1, train_samples_tau2, train_samples_tau3], supertitle="Train", splitview=(3,1))
    fig_val, ax_val, _ = util.plotLineStructures([validate_samples_tau1, validate_samples_tau2, validate_samples_tau3], supertitle="Validation")
    fig_test, ax_test, _ = util.plotLineStructures([test_samples_tau1, test_samples_tau2, test_samples_tau3], supertitle="Test")
    """

    plt.show()
