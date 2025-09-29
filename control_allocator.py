# RBE 577 - Machine Learning for Robotics
# HW 1: Control Allocation via Deep Neural Networks
# ecwenzlaff@wpi.edu

import sys, os
from typing import Any, Tuple, Union # specified types (esp. those using "Any") primarily convey intention and are not guaranteed to work with static type checking libraries 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import vmap_workaround as vw
import matplotlib.pyplot as plt
import random
import plot_utilities as util
import copy

# Global Variables and Parameters:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
F1_bounds = (-1e4, 1e4)
F2_bounds = (-5e3, 5e3)
F3_bounds = (-5e3, 5e3)
alpha2_bounds = (-180.0*(np.pi/180.0), 180.0*(np.pi/180.0))
alpha3_bounds = (-180.0*(np.pi/180.0), 180.0*(np.pi/180.0))
l1, l2, l3, l4 = -14.0, 14.5, -2.7, 2.7
cmd_max = (3e4, 6e4, 6e4, 180.0*(np.pi/180.0), 180.0*(np.pi/180.0))
bad_az_bottom, bad_az_top = (-100.0*(np.pi/180.0), -80.0*(np.pi/180.0)), (80.0*(np.pi/180.0), 100.0*(np.pi/180.0))
k0, k1, k2, k4, k5 = 10e-9, 10e-9, 10e-1, 10e-7, 10e-1    # k3 not included since it's associated with loss function for rate changes which don't apply to this sample set
outputdir = "outputs"

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
    transform_tensor[:,0,1] = torch.cos(forcetensor[3,:].reshape(-1))
    transform_tensor[:,0,2] = torch.cos(forcetensor[4,:].reshape(-1))
    transform_tensor[:,1,0] = 1.0
    transform_tensor[:,1,1] = torch.sin(forcetensor[3,:].reshape(-1))
    transform_tensor[:,1,2] = torch.sin(forcetensor[4,:].reshape(-1))
    transform_tensor[:,2,0] = l2
    transform_tensor[:,2,1] = l1*torch.sin(forcetensor[3,:].reshape(-1)) - l3*torch.cos(forcetensor[3,:].reshape(-1))
    transform_tensor[:,2,2] = l1*torch.sin(forcetensor[4,:].reshape(-1)) - l4*torch.cos(forcetensor[4,:].reshape(-1))
    # Even though I only care about a single slice, broadcasting here could blow up my GPU memory,
    # so I'll need to try using vmap or list comprehension to apply the 5x5 transform to each 5x1 column...
    # torque_tensor = (transform_tensor @ forcetensor)[0,0:3,0:3] 
    def inplacetransform(i: torch.Tensor) -> torch.Tensor:
        return torch.matmul(transform_tensor[i,:,:], forcetensor[:,i])
    torque_tensor = (vw.vmap(inplacetransform)(torch.arange(0,forcetensor.shape[1]))).to(device)
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
        raise(f"splitTrainValTest error - invalid combination of tuple sizes: {len(slicepoints)=}, {len(shuffleset)=}")

class AutoEncoder_LSTM(nn.Module):
    def __init__(self, lstm_nodes: int = 64, lstm_layers: int = 2, net_dropout: float = 0.0):
        super(AutoEncoder_LSTM, self).__init__()
        self.hidden_size = int(lstm_nodes/lstm_layers)
        self.lstm_layers = lstm_layers
        self.encoder_lstm = nn.LSTM(input_size=3, hidden_size=self.hidden_size, num_layers=lstm_layers, dropout=net_dropout).to(device)
        self.encoder_linear = nn.Linear(self.hidden_size, 5).to(device)
        self.decoder_lstm = nn.LSTM(input_size=5, hidden_size=self.hidden_size, num_layers=lstm_layers, dropout=net_dropout).to(device)
        self.decoder_linear = nn.Linear(self.hidden_size, 3).to(device)
    def encode(self, generaltorques: torch.FloatTensor):
        # generaltorques dimension should be (N-by-D) here, where:
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
        # generalforces dimension should be (N-by-D) here, where:
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
        # generaltorques dimension should be (N-by-D) here, where:
        #   N = number of samples (number of batches)
        #   D = torque vector dimension (number of state variables)
        # 
        # As part of each component's respective forward processing, the 2D input tensor 
        # will be expanded to 3D in order to pass it into their respective LSTMs.
        allocatedforces = self.encode(generaltorques)
        decodedtorques = self.decode(allocatedforces)
        return decodedtorques
    
class AutoEncoder_FCN(nn.Module):
    def __init__(self, layer_outputs: Tuple[int, int], net_dropout: float = 0.0):
        super(AutoEncoder_FCN, self).__init__()
        # Since the encoder/decoders are constrained to 3 layers, with a final number 
        # of outputs fixed at 5, the layer_outputs Tuple should always contain two components 
        # ordered for the encoder's hidden layer output sequence (the decoder's hidden layer 
        # output sequence will be the inverse of the encoder layer):
        self.encoder_fcn = nn.Sequential(
            # Input Layer:
            nn.Linear(3, layer_outputs[0]), 
            nn.ReLU(),
            #nn.BatchNorm1d(layer_outputs[0]),
            nn.Dropout(net_dropout),
            # Hidden Layer 1:
            nn.Linear(layer_outputs[0], layer_outputs[1]),
            nn.ReLU(),
            #nn.BatchNorm1d(layer_outputs[1]),
            nn.Dropout(net_dropout),
            # Hidden Layer 2:
            nn.Linear(layer_outputs[1], layer_outputs[2]),
            nn.ReLU(),
            #nn.BatchNorm1d(layer_outputs[2]),
            nn.Dropout(net_dropout),
            # Hidden Layer 3:
            nn.Linear(layer_outputs[2], layer_outputs[3]),
            nn.ReLU(),
            #nn.BatchNorm1d(layer_outputs[3]),
            nn.Dropout(net_dropout),
            # Output Layer:
            nn.Linear(layer_outputs[3], 5)
        ).to(device)
        self.decoder_fcn = nn.Sequential(
            # Input Layer:
            nn.Linear(5, layer_outputs[3]),
            nn.ReLU(),
            #nn.BatchNorm1d(layer_outputs[3]), 
            nn.Dropout(net_dropout),
            # Hidden Layer 1:
            nn.Linear(layer_outputs[3], layer_outputs[2]),
            nn.ReLU(),
            #nn.BatchNorm1d(layer_outputs[2]),
            nn.Dropout(net_dropout),
            # Hidden Layer 2:
            nn.Linear(layer_outputs[2], layer_outputs[1]),
            nn.ReLU(),
            #nn.BatchNorm1d(layer_outputs[1]),
            nn.Dropout(net_dropout),
            # Hidden Layer 3:
            nn.Linear(layer_outputs[1], layer_outputs[0]),
            nn.ReLU(),
            #nn.BatchNorm1d(layer_outputs[0]),
            nn.Dropout(net_dropout),
            # Output Layer:
            nn.Linear(layer_outputs[0], 3)
        ).to(device)
    def encode(self, generaltorques: torch.FloatTensor) -> torch.FloatTensor:
        return self.encoder_fcn(generaltorques)
    def decode(self, generalforces: torch.FloatTensor) -> torch.FloatTensor:
        return self.decoder_fcn(generalforces)
    def forward(self, generaltorques: torch.FloatTensor) -> torch.FloatTensor:
        allocatedforces = self.encode(generaltorques)
        decodedtorques = self.decode(allocatedforces)
        return decodedtorques

def compute_L0(true_torques: torch.FloatTensor, encoder_forces: torch.FloatTensor) -> torch.FloatTensor:
    # This loss function is meant to minimize error between the motion controller's requested torques 
    # and the encoder's commanded generalized forces (after they've been transformed into torques). 
    # true_torques and encoder_forces tensors are expected to have each row represent a unique state 
    # (so that individual vector components have their own column spanning rows equal to the number of 
    # samples to evaluate); so true_torques should be N-by-3, and encoder_forces should be N-by-5:
    encoder_torques = (forcesToTorques(encoder_forces.T).to(device)).T
    error_tensor = (true_torques - encoder_torques[:,0:3]).to(device)
    def MSE(i: torch.Tensor) -> torch.Tensor:
        return torch.mean(error_tensor[i,:]**2)
    mse_tensor = (vw.vmap(MSE)(torch.arange(0,error_tensor.shape[0]))).to(device)   # this should yield a 1 dimensional tensor
    return mse_tensor

def compute_L1(true_torques: torch.FloatTensor, decoder_torques: torch.FloatTensor) -> torch.FloatTensor:
    # This loss function is meant to minimize error between the motion controller's requested torques 
    # and the decoded commanded torques output from the control allocator. true_torques and decoder_forces 
    # tensors are expected to have each row represent a unique state (so that individual vector components 
    # have their own column spanning rows equal to the number of samples to evaluate); so true_torques 
    # should be N-by-3, and decoder_torques should also be N-by-3:
    error_tensor = (true_torques - decoder_torques).to(device)
    def MSE(i: torch.Tensor) -> torch.Tensor:
        return torch.mean(error_tensor[i,:]**2)
    mse_tensor = (vw.vmap(MSE)(torch.arange(0,error_tensor.shape[0]))).to(device)   # this should yield a 1 dimensional tensor
    return mse_tensor

def compute_L2(encoder_forces: torch.FloatTensor) -> torch.FloatTensor:
    # This loss function is meant to minimize the magnitude of the control allocator's commanded forces 
    # based on predefined maximum values associated with each thruster. encoder_forces is expected to 
    # be an N-by-5 tensor, where N is the number of samples:
    sum_tensor = torch.zeros((encoder_forces.shape[0],), dtype=torch.float32).to(device)
    for i in range(0, len(cmd_max)):
        eval_tensor = torch.zeros((encoder_forces.shape[0], 2), dtype=torch.float32).to(device)
        eval_tensor[:,0] = torch.abs(encoder_forces[:,i]) - cmd_max[i]
        sum_tensor += torch.max(eval_tensor, dim=1)[0]
    return sum_tensor

def compute_L3():
    # This loss function was originally meant to penalize large rate changes, but this doesn't apply for this 
    # particular sample dataset
    pass
    
def compute_L4(encoder_forces: torch.FloatTensor) -> torch.FloatTensor:
    # This loss function is meant to minimize power consumption according to Johansen et al's "Constrained
    # nonlinear control allocation with singularity avoidance using sequential quadratic programming".
    # encoder_forces is expected to be an N-by-5 tensor, where N is the number of samples; but only the
    # first 3 columns are used here so it's okay to slice before passing as an input argument:
    power_tensor = torch.zeros((encoder_forces.shape[0],), dtype=torch.float32).to(device)
    power_tensor += torch.abs(encoder_forces[:,0])**(3/2) + \
        torch.abs(encoder_forces[:,1])**(3/2) + \
        torch.abs(encoder_forces[:,2])**(3/2)
    return power_tensor
    
def compute_L5(encoder_forces: torch.FloatTensor) -> torch.FloatTensor:
    # This loss function is meant to minimize the number of instances that the control allocator commands 
    # the thrusters to operate within "inefficient" azimuth sectors. encoder_forces is expected to be an 
    # N-by-5 tensor, where N is the number of samples:
    def check_bottom_zone(i: torch.Tensor) -> torch.Tensor:
        return ( (encoder_forces[:,i] < bad_az_bottom[1]) * (encoder_forces[:,i] > bad_az_bottom[0]) )
    def check_top_zone(i: torch.Tensor) -> torch.Tensor:
        return ( (encoder_forces[:,i] < bad_az_top[1]) * (encoder_forces[:,i] > bad_az_top[0]) )
    bottom_zone = torch.sum(
        (vw.vmap(check_bottom_zone)(torch.tensor([3,4], dtype=int))).reshape(-1,2).to(torch.float32),
        axis=1).to(device)
    top_zone = torch.sum(
        (vw.vmap(check_top_zone)(torch.tensor([3,4], dtype=int))).reshape(-1,2).to(torch.float32),
        axis=1).to(device)
    return (bottom_zone + top_zone).to(device)

def combined_Loss(L0: torch.FloatTensor,
                  L1: torch.FloatTensor,
                  L2: torch.FloatTensor,
                  L4: torch.FloatTensor,
                  L5: torch.FloatTensor) -> torch.FloatTensor:
    # This function calculates the combined loss using the scale factors provided from the paper. 
    # All input loss function tensors are expected to be 1 dimensional with the same number of 
    # elements:
    return ((k0*L0) + (k1*L1) + (k2*L2) + (k4*L4) + (k5*L5)).to(device)

class AutoEncoder_LossFn(nn.Module):
    def __init__(self):
        super(AutoEncoder_LossFn, self).__init__()
    def forward(self, true_torques: torch.FloatTensor, encoder_forces: torch.FloatTensor, decoder_torques: torch.FloatTensor) -> torch.FloatTensor:
        loss0 = compute_L0(true_torques, encoder_forces)
        loss1 = compute_L1(true_torques, decoder_torques)
        loss2 = compute_L2(encoder_forces)
        loss4 = compute_L4(encoder_forces)
        loss5 = compute_L5(encoder_forces)
        return torch.mean(combined_Loss(loss0, loss1, loss2, loss4, loss5)) # safe to take mean since L0 -> L5 were designed to yield only positive values

def normalizeTorques(torque_tensor: torch.FloatTensor) -> torch.FloatTensor:
    mu = torch.mean(torque_tensor, axis=0).reshape(1,-1).to(device)
    sigma = torch.sqrt((1/torque_tensor.shape[0])*(torch.sum((torque_tensor - mu)**2, dim=0))).to(device)
    #sigma2 = torch.std(torque_tensor, dim=0)
    return ((torque_tensor - mu)/sigma).to(device)

if __name__ == '__main__':
    # NOTE: we don't need to recreate figures 9, 10, and 5
    print(f"Python Version = {sys.version}, Torch Device = {device}")
    # Generate datasets:
    test_tensor = randomWalkTensor(
        (5,int(1e6)), 
        torch.tensor([100.0, 10.0, 10.0, 2.0*(np.pi/180.0), 2.0*(np.pi/180.0)]).reshape(-1,1),
        torch.tensor([
            random.uniform(F1_bounds[0], F1_bounds[1]),
            random.uniform(F2_bounds[0], F2_bounds[1]),
            random.uniform(F3_bounds[0], F3_bounds[1]),
            random.uniform(alpha2_bounds[0], alpha2_bounds[1]),
            random.uniform(alpha3_bounds[0], alpha3_bounds[1])
        ]).reshape(-1)).to(device)
    torque_tensor = forcesToTorques(test_tensor)
    T_train, T_validation, T_test = splitTrainValTest(torque_tensor.cpu().detach().numpy(), (0.7, 0.8), (True, False, False), 1)
    
    # Configure the NN model and training parameters:
    #model = AutoEncoder_LSTM()
    model = AutoEncoder_FCN((60, 90, 30, 10), net_dropout=0.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    loss_fn = AutoEncoder_LossFn()
    num_epochs = 100
    batch_size = 2000
    batch_start = torch.arange(0, T_train.shape[1], batch_size)
    best_loss = np.inf
    best_modelstate = None
    training_log, training_epoch_log, validation_log, test_log = [], [], [], []     # each list entry should be a sub-list with losses: L0, L1, L2, L4, L5, Lcombined 

    # Training Loop:
    for epoch in range(0, num_epochs):
        last_loss_this_epoch = np.inf
        model.train()
        print(f"Epoch {epoch+1}/{num_epochs}")
        for start in batch_start:
            # Take a batch and perform forward pass:
            T_batch = T_train[:,start:(start+batch_size)].T
            model_outputs = model(T_batch) #T_batch) #normalizeTorques(T_batch))
            encoder_outputs = model.encode(T_batch) #T_batch) #normalizeTorques(T_batch))
            loss0 = torch.mean(compute_L0(T_batch, encoder_outputs))
            loss1 = torch.mean(compute_L1(T_batch, model_outputs))
            loss2 = torch.mean(compute_L2(encoder_outputs))
            loss4 = torch.mean(compute_L4(encoder_outputs))
            loss5 = torch.mean(compute_L5(encoder_outputs))
            #print(f"{loss0=}\n {loss1=}\n {loss2=}\n {loss4=}\n {loss5=}")
            loss_comb = loss_fn(T_batch, encoder_outputs, model_outputs)
            last_loss_this_epoch = loss_comb.item()

            # Store losses:
            training_log.append([loss0.item(), loss1.item(), loss2.item(), loss4.item(), loss5.item(), loss_comb.item()])

            # Perform backward pass:
            optimizer.zero_grad()
            loss_comb.backward()

            # Update weights and print progress:
            #nn.utils.clip_grad_norm_(model.parameters(), 1e-3)
            optimizer.step()
            print(f"\tepoch: {epoch+1}/{num_epochs}, Batch {start//batch_size+1}/{len(batch_start)}, Loss: {loss_comb.item()}")
        
        # At the end of each epoch, perform evaluation with the validation data:
        training_epoch_log.append(last_loss_this_epoch)
        model.eval()
        with torch.no_grad():
            model_outputs = model(T_validation.T) #T_validation.T) #normalizeTorques(T_validation.T))
            encoder_outputs = model.encode(T_validation.T) #T_validation.T) #normalizeTorques(T_validation.T))
            loss0 = torch.mean(compute_L0(T_validation.T, encoder_outputs))
            loss1 = torch.mean(compute_L1(T_validation.T, model_outputs))
            loss2 = torch.mean(compute_L2(encoder_outputs))
            loss4 = torch.mean(compute_L4(encoder_outputs))
            loss5 = torch.mean(compute_L5(encoder_outputs))
            loss_comb = loss_fn(T_validation.T, encoder_outputs, model_outputs)
            validation_log.append([loss0.item(), loss1.item(), loss2.item(), loss4.item(), loss5.item(), loss_comb.item()])
            if (loss_comb.item() < best_loss):
                best_loss = loss_comb.item()
                best_modelstate = copy.deepcopy(model.state_dict())
        scheduler.step(loss_comb)

    # Once the training is complete, restore model to its "best state" and perform evaluation against test data:
    #print(f"{best_modelstate = }")
    print(f"{best_loss = :.2f}")
    model.load_state_dict(best_modelstate)
    model.eval()
    final_model_outputs = model(T_test.T) #T_test.T) #normalizeTorques(T_test.T))
    final_encoder_outputs = model.encode(T_test.T) #T_test.T) #normalizeTorques(T_test.T))
    final_decoded_torques = (forcesToTorques(final_encoder_outputs.T)).T
    final_loss0 = compute_L0(T_test.T, final_encoder_outputs)
    final_loss1 = compute_L1(T_test.T, final_model_outputs)
    final_loss2 = compute_L2(final_encoder_outputs)
    final_loss4 = compute_L4(final_encoder_outputs)
    final_loss5 = compute_L5(final_encoder_outputs)
    final_loss_comb = combined_Loss(final_loss0, final_loss1, final_loss2, final_loss4, final_loss5)

    test_tau1 = util.LineStructure(x=np.arange(0, T_test.shape[1]), y=T_test.cpu().detach().numpy()[0,:].reshape(-1,1), label="Tau1")
    test_tau2 = util.LineStructure(x=np.arange(0, T_test.shape[1]), y=T_test.cpu().detach().numpy()[1,:].reshape(-1,1), label="Tau2")
    test_tau3 = util.LineStructure(x=np.arange(0, T_test.shape[1]), y=T_test.cpu().detach().numpy()[2,:].reshape(-1,1), label="Tau3")
    test_F1 = util.LineStructure(x=np.arange(0, final_encoder_outputs.shape[0]), y=(final_encoder_outputs.cpu().detach().numpy()[:,0]).reshape(-1,1), label="F1")
    test_F2 = util.LineStructure(x=np.arange(0, final_encoder_outputs.shape[0]), y=(final_encoder_outputs.cpu().detach().numpy()[:,1]).reshape(-1,1), label="F2")
    test_F3 = util.LineStructure(x=np.arange(0, final_encoder_outputs.shape[0]), y=(final_encoder_outputs.cpu().detach().numpy()[:,2]).reshape(-1,1), label="F3")
    test_a2 = util.LineStructure(x=np.arange(0, final_encoder_outputs.shape[0]), y=((180.0/np.pi)*final_encoder_outputs.cpu().detach().numpy()[:,3]).reshape(-1,1), label="alpha2")
    test_a3 = util.LineStructure(x=np.arange(0, final_encoder_outputs.shape[0]), y=((180.0/np.pi)*final_encoder_outputs.cpu().detach().numpy()[:,4]).reshape(-1,1), label="alpha3")
    dec_tau1 = util.LineStructure(x=np.arange(0, final_decoded_torques.shape[0]), y=(final_decoded_torques.cpu().detach().numpy()[:,0]).reshape(-1,1), label="Tau1 (T*encoder)")
    dec_tau2 = util.LineStructure(x=np.arange(0, final_decoded_torques.shape[0]), y=(final_decoded_torques.cpu().detach().numpy()[:,1]).reshape(-1,1), label="Tau2 (T*encoder)")
    dec_tau3 = util.LineStructure(x=np.arange(0, final_decoded_torques.shape[0]), y=(final_decoded_torques.cpu().detach().numpy()[:,2]).reshape(-1,1), label="Tau3 (T*encoder)")
    train_loss_s = util.LineStructure(x=np.arange(0, len(training_log)), y=np.array(training_log)[:,-1])
    epoch_train_loss = util.LineStructure(x=np.arange(0, len(training_epoch_log)), y=np.array(training_epoch_log), label="Training")
    val_loss_s = util.LineStructure(x=np.arange(0, len(validation_log)), y=np.array(validation_log)[:,-1], label="Validation")
    test_loss_s = util.LineStructure(x=np.arange(0, final_loss_comb.shape[0]), y=final_loss_comb.cpu().detach().clone().numpy())
    test_loss0 = util.LineStructure(x=np.arange(0, final_loss0.shape[0]), y=final_loss0.cpu().detach().clone().numpy())
    test_loss1 = util.LineStructure(x=np.arange(0, final_loss1.shape[0]), y=final_loss1.cpu().detach().clone().numpy())
    test_loss2 = util.LineStructure(x=np.arange(0, final_loss2.shape[0]), y=final_loss2.cpu().detach().clone().numpy())
    test_loss4 = util.LineStructure(x=np.arange(0, final_loss4.shape[0]), y=final_loss4.cpu().detach().clone().numpy())
    test_loss5 = util.LineStructure(x=np.arange(0, final_loss5.shape[0]), y=final_loss5.cpu().detach().clone().numpy())
    
    tau_fig, tau_ax, _ = util.plotLineStructures([test_tau1, test_tau2, test_tau3], supertitle="Test Taus", xlabels=["Samples"], ylabels=["Torque [N*m]"])
    forces_fig, forces_ax, _ = util.plotLineStructures([test_F1, test_F2, test_F3], supertitle="Test Encoder Forces", xlabels=["Samples"], ylabels=["Force [N]"])
    transform1_fig, transform1_ax, _ = util.plotLineStructures([test_tau1, dec_tau1], supertitle="Test Input Tau1 vs Encoder Output", xlabels=["Samples"], ylabels=["Torque [N*m]"])
    transform2_fig, transform2_ax, _ = util.plotLineStructures([test_tau2, dec_tau2], supertitle="Test Input Tau2 vs Encoder Output", xlabels=["Samples"], ylabels=["Torque [N*m]"])
    transform3_fig, transform3_ax, _ = util.plotLineStructures([test_tau3, dec_tau3], supertitle="Test Input Tau3 vs Encoder Output", xlabels=["Samples"], ylabels=["Torque [N*m]"])
    angles_fig, angles_ax, _ = util.plotLineStructures([test_a2, test_a3], supertitle="Test Encoder Angles", xlabels=["Samples"], ylabels=["Angle [degrees]"])
    train_fig, train_ax, _ = util.plotLineStructures([train_loss_s], supertitle="Combined Training Loss", xlabels=["Mini Batches"], ylabels=["Loss"])
    trval_fig, trval_ax, _ = util.plotLineStructures([epoch_train_loss, val_loss_s], supertitle="Combined Training vs Validation Loss", xlabels=["Epoch"], ylabels=["Loss"])
    test_fig, test_ax, _ = util.plotLineStructures([test_loss_s], supertitle="Combined Test Loss", xlabels=["Samples"], ylabels=["Loss"])
    losses_fig, losses_ax, _ = util.plotLineStructures([test_loss0, test_loss1, test_loss2, test_loss4, test_loss5], supertitle="Test Loss Components", splitview=(5,1), 
                                                       xlabels=5*["Samples"], ylabels=["L0 Loss", "L1 Loss", "L2 Loss", "L4 Loss", "L5 Loss"])

    if (not os.path.exists(outputdir)):
        os.makedirs(outputdir)
    tau_fig.savefig(f"{outputdir}/tau_fig.png")
    forces_fig.savefig(f"{outputdir}/forces_fig.png")
    transform1_fig.savefig(f"{outputdir}/transform1_fig.png")
    transform2_fig.savefig(f"{outputdir}/transform2_fig.png")
    transform3_fig.savefig(f"{outputdir}/transform3_fig.png")
    angles_fig.savefig(f"{outputdir}/angles_fig.png")
    train_fig.savefig(f"{outputdir}/train_fig.png")
    trval_fig.savefig(f"{outputdir}/trval_fig.png")
    test_fig.savefig(f"{outputdir}/test_fig.png")
    losses_fig.set_size_inches(8.0, 12.0)
    losses_fig.savefig(f"{outputdir}/losses_fig.png")
    with open(f"{outputdir}/best_model_params.txt","w") as txtout:
        print(f"{best_modelstate}", file=txtout)

    plt.show()
