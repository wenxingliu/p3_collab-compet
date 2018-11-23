import numpy as np
from collections import deque
import torch

from unityagents import UnityEnvironment
from agent import Agent
from utils import plot_scores
from hyperparams import *


def main(episodes, print_every=100):

    env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis")
    agent = Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE)

    scores = []
    scores_window = deque(maxlen=100)

    for ep in range(1, int(episodes) + 1):
        agent.reset(env)
        game_end = False

        while not game_end:
            agent.step(env, add_noise=True)
            game_end = any(agent.dones)

        scores.append(np.max(agent.scores))
        scores_window.append(np.max(agent.scores))

        if ep % print_every == 0:
            print('Episode %d, score: %.2f' % (ep, np.max(agent.scores)))

        if np.mean(scores_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(ep - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor.state_dict(), 'checkpoints//tennis_actor_checkpoint.pth')
            torch.save(agent.critic.state_dict(), 'checkpoints//tennis_critic_checkpoint.pth')

    plot_scores(scores)
    # pickle.dump(agent, open('checkpoints//tennis_agent.p', 'w'))
    env.close()


if __name__ == '__main__':
    main(episodes=1e5)