"""SAC の実装"""

from tqdm.auto import trange
import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from env import CustomEnv
from memory import ReplayMemory
from network import GaussianPolicyNet, QFunctionNet
from util import ScoreLogger, evaluate_policy, soft_update, action_converter_for_pendulum, log_prob_of_squashed_action


class AgentConfig:
    lr_policy = 5e-3
    lr_q = 1e-2
    lr_alpha = 1e-2
    gamma = 0.98
    memory_size = 10000
    batch_size = 32
    min_num_experience = 2000
    tau = 0.01
    initial_alpha = 0.01


class Agent(AgentConfig):
    def __init__(self, state_size, action_size, action_converter):
        self.policy = GaussianPolicyNet(state_size)

        self.q1 = QFunctionNet(state_size)
        self.q2 = QFunctionNet(state_size)
        self.q1_target = QFunctionNet(state_size)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target = QFunctionNet(state_size)
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.objective_entropy = -action_size
        self.log_alpha = torch.tensor(np.log(self.initial_alpha), requires_grad=True)

        self.optimizer_policy = optim.Adam(self.policy.parameters(), self.lr_policy)
        self.optimizer_q = optim.Adam([
            {'params': self.q1.parameters(), 'lr': self.lr_q},
            {'params': self.q2.parameters(), 'lr': self.lr_q},
        ])
        self.optimizer_alpha = optim.Adam([self.log_alpha], self.lr_alpha)

        self.memory = ReplayMemory(self.memory_size, self.batch_size)

        self.action_converter = action_converter

    def get_action(self, state, explore=False):
        with torch.no_grad():
            mu, sigma = self.policy(state)
            if not explore:
                action = mu
            pi = Normal(mu, sigma)
            action = pi.sample()

        return self.action_converter(action)

    def update(self):
        if len(self.memory) < self.min_num_experience:
            return

        states, actions, rewards, next_states, dones = self.memory.get_batch()

        alpha = self.log_alpha.exp().detach()

        q1s = self.q1(states, actions)
        q2s = self.q2(states, actions)

        target_mus, target_sigmas = self.policy(next_states)
        target_pis = Normal(target_mus, target_sigmas)
        target_pre_squash_actions = target_pis.sample()
        target_entropies = log_prob_of_squashed_action(target_pis, target_pre_squash_actions).detach()

        target_actions = self.action_converter(target_pre_squash_actions)
        target_q1s = self.q1_target(next_states, target_actions).detach()
        target_q2s = self.q2_target(next_states, target_actions).detach()
        td_targets = rewards + self.gamma * (1 - dones) * (torch.min(target_q1s, target_q2s) + alpha * target_entropies)

        loss_q1 = F.mse_loss(q1s, td_targets)
        loss_q2 = F.mse_loss(q2s, td_targets)

        self.optimizer_q.zero_grad()
        loss_q1.backward()
        loss_q2.backward()
        self.optimizer_q.step()

        mu, sigma = self.policy(states)
        pis = Normal(mu, sigma)
        reparameterize_pre_squash_actions = pis.rsample()
        reparameterize_entropies = log_prob_of_squashed_action(pis, reparameterize_pre_squash_actions)

        reparameterize_actions = self.action_converter(reparameterize_pre_squash_actions)
        q1s = self.q1(states, reparameterize_actions)
        q2s = self.q2(states, reparameterize_actions)

        loss_policy = -(torch.min(q1s, q2s) + alpha * reparameterize_entropies).mean()

        self.optimizer_policy.zero_grad()
        loss_policy.backward()
        self.optimizer_policy.step()

        loss_alpha = (self.log_alpha.exp() * (reparameterize_entropies - self.objective_entropy).detach()).mean()
        self.optimizer_alpha.zero_grad()
        loss_alpha.backward()
        self.optimizer_alpha.step()

        soft_update(self.q1_target, self.q1, self.tau)
        soft_update(self.q2_target, self.q2, self.tau)


def main():
    env_name = 'Pendulum-v1'
    state_size = 3
    action_size = 1
    action_shape = (1)
    max_episode_steps = 200
    env = CustomEnv(gym.make(env_name), max_episode_steps)

    agent = Agent(state_size, action_size, action_converter_for_pendulum)

    episodes = 1000
    dump_interval = 50

    logger = ScoreLogger(dump_interval)

    progress = trange(1, episodes + 1, dynamic_ncols=True)
    for episode in progress:
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state, explore=True)
            next_state, reward, done = env.step(action.numpy().reshape(action_shape))
            agent.memory.add_experience(state, action, reward, next_state, done)
            state = next_state

        for _ in range(20):
            agent.update()

        logger.log(evaluate_policy(agent, CustomEnv(gym.make(env_name), max_episode_steps), action_shape))

        if episode % dump_interval == 0:
            progress.set_postfix_str(f"avg_steps = {logger.average_score_log[-1]:.2f}")

    logger.plot()

if __name__ == '__main__':
    main()
