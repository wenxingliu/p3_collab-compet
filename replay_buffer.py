from collections import deque, namedtuple
import numpy as np
import random
import torch

from utils import array_to_tensor

device = "cuda" if torch.cuda.is_available() else "cpu"


class Memory:

    def __init__(self, buffer_size, batch_size, num_agents=2):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(0)

    def add(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            experience = self.experience(states[i], actions[i], rewards[i], next_states[i], dones[i])
            self.memory.append(experience)

    def sample(self):
        sampled_experiences = random.sample(self.memory, k=self.batch_size)

        states = np.vstack([exp.state for exp in sampled_experiences])
        actions = np.vstack([exp.action for exp in sampled_experiences])
        rewards = np.vstack([exp.reward for exp in sampled_experiences])
        next_states = np.vstack([exp.next_state for exp in sampled_experiences])
        dones = np.vstack([exp.done for exp in sampled_experiences]).astype(np.uint8)

        states = array_to_tensor(states)
        actions = array_to_tensor(actions)
        rewards = array_to_tensor(rewards)
        next_states = array_to_tensor(next_states)
        dones = array_to_tensor(dones)

        return states, actions, rewards, next_states, dones

    def has_enough_memory(self):
        return len(self.memory) >= self.batch_size