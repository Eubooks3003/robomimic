import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
class TrajectoryClassifier(nn.Module):
    def __init__(self, state_dim, action_dim, num_past, num_future):
        super(TrajectoryClassifier, self).__init__()
        
        self.num_past = num_past
        self.num_future = num_future
        # Calculate the input dimension for the MLP (flatten state-action pairs across time horizon
        input_dim = (num_past + 1) * state_dim + (num_past + num_future + 1) * state_dim

        print("Input Dim: ", input_dim)
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Flatten the input across the state, action, and time horizon dimensions
        x = x.contiguous().reshape(x.size(0), -1)  # [batch_size, (state_dim + action_dim) * time_horizon]
        
        # Forward pass through fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        
        # Output layer
        x = self.fc4(x)
        x = self.sigmoid(x)
        
        return x

    def get_num_past(self):
        return self.num_past
    def get_num_future(self):
        return self.num_future

class MultiTrajectoryDataset(Dataset):
    def __init__(self, all_states, all_actions, all_results, trajectory_lengths, num_past=5, num_future=5, mode='train'):
        self.all_states = all_states
        self.all_actions = all_actions
        self.all_results = all_results
        self.trajectory_lengths = trajectory_lengths
        self.num_past = num_past
        self.num_future = num_future
        self.mode = mode
        self.indices = self._generate_indices()

    def _generate_indices(self):
        indices = []
        start_idx = 0
        
        for traj_idx, traj_length in enumerate(self.trajectory_lengths):
            for i in range(self.num_past, traj_length - self.num_future):
                indices.append((traj_idx, start_idx + i))
            start_idx += traj_length

        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, current_idx = self.indices[idx]
        
        # Adjust number of past and future pairs based on the mode
        if self.mode == 'inference':
            num_past = self.num_past + self.num_future
            num_future = 0
        else:
            num_past = self.num_past
            num_future = self.num_future

        # Calculate the start and end indices for the window
        window_start_idx = max(0, current_idx - num_past)
        window_end_idx = current_idx + 1  # Include current observation

        # Get past observations + current observation
        states_window = self.all_states[window_start_idx:window_end_idx]
        current_state = torch.tensor(self.all_states[current_idx], dtype=torch.float32)  

        # Get actions: past actions + current action + future actions
        action_start_idx = max(0, current_idx - num_past)
        action_end_idx = current_idx + num_future + 1
        actions_window = self.all_actions[action_start_idx:action_end_idx]

        # Padding actions to match the observation dimensions
        padded_actions = []
        for action in actions_window:
            action_pad = torch.zeros_like(current_state)  # Match observation dimension
            action_pad[:len(action)] = torch.tensor(action, dtype=torch.float32)
            padded_actions.append(action_pad)
        
        # Convert windows to tensors
        states_tensor = torch.tensor(states_window, dtype=torch.float32)

        actions_tensor = torch.stack(padded_actions, dim=0)

        states_tensor = states_tensor.transpose(0, 1)
        actions_tensor = actions_tensor.transpose(0, 1)

        # Concatenate states and padded actions
        state_action_seq = torch.cat((states_tensor, actions_tensor), dim=-1)

        # Get the result associated with the trajectory
        result = self.all_results[traj_idx]
        result_tensor = torch.tensor(result, dtype=torch.float32)  # Assuming result is a scalar

        return state_action_seq, result_tensor
