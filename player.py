import numpy as np
import torch

from utils import array_to_tensor
from ddpg_nets import OUNoise


class Players:

    def __init__(self, actor, action_size, state_size, max_steps=100):
        self.actor = actor
        self.noise = OUNoise(self.action_size)
        self.action_size = action_size
        self.state_size = state_size
        self.max_steps = max_steps

        self.step_count, self.score = 0, 0
        self.state, self.action, self.reward, self.next_state, self.done = None, None, None, None, None

    def reset(self):
        self.noise.reset()
        self.step_count, self.score = 0, 0
        self.state, self.action, self.reward, self.next_state, self.done = None, None, None, None, None

    def act(self, add_noise=True):
        state = array_to_tensor(self.state)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
            action = action.cpu().data.numpy()
        self.actor.train()

        if add_noise:
            noise = self.noise.sample()
            action += noise

        action = np.clip(action, -1, 1)
        return action