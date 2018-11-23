import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

device = "cuda" if torch.cuda.is_available() else "cpu"


def seeding(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)

def array_to_tensor(arr):
    return torch.from_numpy(arr).float().to(device)


def hard_update(local_netowrks, target_networks):
    for target_param, local_param in zip(target_networks.parameters(), local_netowrks.parameters()):
        target_param.data.copy_(local_param.data)


def soft_update(local_netowrks, target_networks, tau):
    for target_param, local_param in zip(target_networks.parameters(), local_netowrks.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def unpack_trajectories(trajectories):
    states = array_to_tensor(np.array([trajectory.states for trajectory in trajectories]))
    actions = array_to_tensor(np.array([trajectory.actions for trajectory in trajectories]))
    rewards = array_to_tensor(np.array([trajectory.rewards for trajectory in trajectories]))
    next_states = array_to_tensor(np.array([trajectory.next_states for trajectory in trajectories]))
    dones = np.array([trajectory.dones for trajectory in trajectories])
    return states, actions, rewards, next_states, dones


def ddpg_compute_critic_loss(states, actions, rewards, next_states, dones,
                             target_actor, target_critic, local_critic):
    actions_next = target_actor(next_states)
    target_q_next = target_critic(next_states, actions_next)
    # Compute Q targets for current states (y_i)
    target_q = rewards + (target_q_next * (1 - dones))
    # Compute critic loss
    local_q_pred = local_critic(states, actions)
    critic_loss = F.mse_loss(local_q_pred, target_q)
    return critic_loss


def ddpg_compute_actor_loss(states, local_actor, local_critic):
    actions_pred = local_actor(states)
    actor_loss = - local_critic(states, actions_pred).mean()
    return actor_loss


def test_agent(env, agent, test_episodes=10):
    env.train_mode = False
    scores = []
    for i in range(1, test_episodes+1):

        agent.reset()
        env.reset()
        agent.states = env.reset()
        done = False
        while not done:
            agent.act(add_noise=False)
            agent.rewards, agent.next_states, agent.dones = env.step(agent.actions)
            agent.scores += agent.rewards
            agent.step_count += 1
            agent.states = agent.next_states
            done = any(agent.dones)

        print('Episode %d, avg score %.2f' % (i, agent.scores.mean()))

        scores.append(agent.scores.mean())

    return scores


def plot_scores(scores, title='DDPG'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.show()

    fig.savefig('%s_scores.png' % title)