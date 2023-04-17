import random
import pickle
from collections import namedtuple, deque

import numpy as np
import torch
from torch_geometric.data import Data, Batch


Experience = namedtuple(
    "Experience", field_names=[
        "state", "choice", "reward", "next_state",
        "next_legal_choices", "finished", "moving"
    ]
)


class ReplayBuffer:
    def __init__(self, choice_size, batch_size, buffer_size, device):
        self.choice_size = choice_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.device = device

    def add(self, experience):
        self.memory.append(Experience(*experience))

    def sample(self):
        states, choices, rewards, next_states, next_legal_choices, finished, moving = zip(
            *random.sample(self.memory, k=self.batch_size)
        )

        if isinstance(states[0], np.ndarray):
            states = torch.tensor(
                states, dtype=torch.float32, device=self.device
            )
            next_states = torch.tensor(
                next_states, dtype=torch.float32, device=self.device
            )
        elif isinstance(states[0], Data):
            states = Batch.from_data_list(states).to(self.device)
            next_states = Batch.from_data_list(next_states).to(self.device)

        choices = torch.tensor(
            choices, dtype=torch.int64, device=self.device
        )
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        next_legal_choices = torch.tensor(
            next_legal_choices, dtype=torch.bool, device=self.device
        )
        finished = torch.tensor(
            finished, dtype=torch.uint8, device=self.device
        )
        moving = torch.tensor(
            moving, dtype=torch.bool, device=self.device
        )

        return states, choices, rewards, next_states, next_legal_choices, finished, moving

    def can_sample(self):
        
        return len(self.memory) >= self.batch_size

    def save(self, filename):

        with open(filename, 'wb') as f:
            pickle.dump(list(self.memory), f)

    def load(self, filename):

        with open(filename, 'rb') as f:
            self.memory = pickle.load(f)

    def __len__(self):
        return len(self.memory)
