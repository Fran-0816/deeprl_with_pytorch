"""マルチステップ学習の実装"""

import random
from tqdm.auto import trange
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim

from memory import MultiStepReplayMemory, TransitionMemory
from env import CustomEnv
from network import DQN
from util import ScoreLogger, evaluate_policy, hard_update


class AgentConfig:
    lr = 2e-4
    epsilon = 0.3
    gamma = 0.98
    memory_size = 10000
    batch_size = 32
    min_num_experience = 2000
    n_steps = 3


class Agent(AgentConfig):
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.q = DQN(state_size, action_size)
        self.q_target = DQN(state_size, action_size)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)

        self.experience = MultiStepReplayMemory(self.memory_size, self.batch_size)
        self.trajectory = TransitionMemory()

    def get_action(self, state, explore=False):
        if explore and random.random() < self.epsilon:
            return torch.tensor([random.randrange(self.action_size)])
        else:
            with torch.no_grad():
                qs = self.q(state)
                action = qs.argmax(dim=-1, keepdim=True)
                return action

    def update(self):
        if len(self.experience) < self.min_num_experience:
            return

        states, actions, td_targets = self.experience.get_batch()

        qs = self.q(states)
        qs = qs.gather(dim=1, index=actions)

        loss = F.mse_loss(qs, td_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def final_path(self):
        steps = len(self.trajectory)

        states, actions, rewards, next_states, dones = self.trajectory.get_batch()

        rewards = torch.cat((rewards, torch.zeros((self.n_steps - 1, 1))), dim=0)
        dones = torch.cat((dones, torch.ones((self.n_steps, 1), dtype=torch.long)), dim=0)

        target_qs = self.q_target(next_states[self.n_steps:]).detach()
        target_qs = target_qs.max(dim=1, keepdim=True)[0]
        target_qs = torch.cat((target_qs, torch.zeros((self.n_steps, 1))), dim=0)

        td_targets = torch.zeros((steps, 1))
        for t in range(self.n_steps):
            td_targets = td_targets + rewards[t:t+steps]
            rewards *= self.gamma
        td_targets = td_targets + (1 - dones[self.n_steps:]) * torch.tensor(self.gamma)**self.n_steps * target_qs

        for state, action, td_target in zip(states, actions, td_targets):
            self.experience.add_experience(state, action, td_target)


def main():
    env_name = 'CartPole-v1'
    state_size = 4
    action_size = 2
    action_shape = ()
    max_episode_steps = 500
    env = CustomEnv(gym.make(env_name), max_episode_steps)

    agent = Agent(state_size, action_size)

    episodes = 2500
    sync_interval = 20
    dump_interval = 50

    logger = ScoreLogger(dump_interval)

    progress = trange(1, episodes + 1, dynamic_ncols=True)
    for episode in progress:
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state, explore=True)
            next_state, reward, done = env.step(action.numpy().reshape(action_shape))
            agent.trajectory.add_transition(state, action, reward, next_state, done)
            state = next_state

        agent.final_path()

        for _ in range(20):
            agent.update()

        logger.log(evaluate_policy(agent, CustomEnv(gym.make(env_name), max_episode_steps), action_shape))

        if episode % sync_interval == 0:
            hard_update(agent.q_target, agent.q)

        if episode % dump_interval == 0:
            progress.set_postfix_str(f"avg_steps = {logger.average_score_log[-1]:.2f}")

    logger.plot()

if __name__ == '__main__':
    main()
