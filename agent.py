import torch
import torch.optim as optim
import numpy as np

from hyperparams import *
from ddpg_nets import Actor, Critic, OUNoise
from replay_buffer import Memory
from utils import soft_update, hard_update, ddpg_compute_actor_loss, ddpg_compute_critic_loss, array_to_tensor


class Agent:

    def __init__(self, state_size, action_size, num_agents=2, brain_name='TennisBrain'):
        self.state_size = state_size
        self.action_size = action_size
        self.brain_name = brain_name
        self.num_agents = num_agents

        self.scores = np.zeros(self.num_agents)
        self.step_count = 0
        self.states, self.actions, self.rewards, self.next_states = None, None, None, None

        self.memory = Memory(batch_size=BATCH_SIZE, buffer_size=MEMORY_BUFFER)
        self.noise = OUNoise(size=ACTION_SIZE)

        self.actor = Actor(state_size=STATE_SIZE, action_size=ACTION_SIZE)
        self.critic = Critic(state_size=STATE_SIZE, action_size=ACTION_SIZE)

        self.target_actor = Actor(state_size=STATE_SIZE, action_size=ACTION_SIZE)
        self.target_critic = Critic(state_size=STATE_SIZE, action_size=ACTION_SIZE)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        hard_update(self.actor, self.target_actor)
        hard_update(self.critic, self.target_critic)

    def reset(self, env):
        self.noise.reset()
        self.scores = np.zeros(self.num_agents)
        self.step_count = 0
        self.states, self.actions, self.rewards, self.next_states = None, None, None, None

        env_info = env.reset()[self.brain_name]
        self.states = env_info.vector_observations

    def act(self, add_noise=True):
        states = array_to_tensor(self.states)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(states)
            actions = actions.cpu().data.numpy()
        self.actor.train()

        if add_noise:
            action_noise = self.noise.sample()
            actions += action_noise

        self.actions = np.clip(actions, -1, 1)

    def step(self, env, add_noise=True):
        self.act(add_noise)
        env_info = env.step(self.actions)[self.brain_name]
        self.next_states = env_info.vector_observations
        self.rewards = env_info.rewards
        self.dones = env_info.local_done

        self.memory.add(self.states, self.actions, self.rewards, self.next_states, self.dones)

        self.states = self.next_states
        self.scores += self.rewards
        self.step_count += 1

        if self.memory.has_enough_memory():
            for i in range(UPDATE_FREQUENCY_PER_STEP):
                states, actions, rewards, next_states, dones = self.memory.sample()
                self.learn(states, actions, rewards, next_states, dones)

    def learn(self, states, actions, rewards, next_states, dones):
        # Update critic
        self.critic_opt.zero_grad()
        critic_loss = ddpg_compute_critic_loss(states, actions, rewards, next_states, dones,
                                               self.target_actor, self.target_critic, self.critic)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_opt.step()

        # Update actor
        self.actor_opt.zero_grad()
        actor_loss = ddpg_compute_actor_loss(states, self.actor, self.critic)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_opt.step()

        # Update target nets
        soft_update(self.actor, self.target_actor, TAU)
        soft_update(self.critic, self.target_critic, TAU)