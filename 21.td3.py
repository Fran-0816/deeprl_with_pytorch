"""TD3 の実装"""

from tqdm.auto import trange
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim

from env import CustomEnv
from memory import ReplayMemory
from network import DeterministicPolicyNet, QFunctionNet
from util import ScoreLogger, evaluate_policy, soft_update, action_converter_for_pendulum


class AgentConfig:
    lr_policy = 2e-3
    lr_q = 1e-2
    gamma = 0.98
    memory_size = 10000
    batch_size = 32
    min_num_experience = 2000
    tau = 0.01
    sigma = 1
    sigma_tilde = 0.2
    c = 0.5


class Agent(AgentConfig):
    def __init__(self, state_size):
        self.policy = DeterministicPolicyNet(state_size, action_converter_for_pendulum)
        self.policy_target = DeterministicPolicyNet(state_size, action_converter_for_pendulum)
        self.policy_target.load_state_dict(self.policy.state_dict())

        self.q1 = QFunctionNet(state_size)
        self.q2 = QFunctionNet(state_size)
        self.q1_target = QFunctionNet(state_size)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target = QFunctionNet(state_size)
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.optimizer_policy = optim.Adam(self.policy.parameters(), self.lr_policy)
        self.optimizer_q = optim.Adam([
            {'params': self.q1.parameters(), 'lr': self.lr_q},
            {'params': self.q2.parameters(), 'lr': self.lr_q},
        ])

        self.memory = ReplayMemory(self.memory_size, self.batch_size)

    @torch.no_grad()
    def get_action(self, state, explore=False):
        action = self.policy(state)
        if explore:
            action = action + torch.randn(1) * self.sigma
            action = torch.clip(action, -2, 2)

        return action

    def update(self, update_policy):
        if len(self.memory) < self.min_num_experience:
            return

        states, actions, rewards, next_states, dones = self.memory.get_batch()

        q1s = self.q1(states, actions)
        q2s = self.q2(states, actions)

        target_actions = torch.clip(self.policy_target(next_states) + torch.clip(torch.randn(1) * self.sigma_tilde, -self.c, self.c), -2, 2)
        target_q1s = self.q1_target(next_states, target_actions).detach()
        target_q2s = self.q2_target(next_states, target_actions).detach()
        td_targets = rewards + self.gamma * (1 - dones) * torch.min(target_q1s, target_q2s)

        loss_q1 = F.mse_loss(q1s, td_targets.detach())
        loss_q2 = F.mse_loss(q2s, td_targets.detach())

        self.optimizer_q.zero_grad()
        loss_q1.backward()
        loss_q2.backward()
        self.optimizer_q.step()

        if update_policy:
            loss_policy = -self.q1(states, self.policy(states)).mean()

            self.optimizer_policy.zero_grad()
            loss_policy.backward()
            self.optimizer_policy.step()

            soft_update(self.policy_target, self.policy, self.tau)
            soft_update(self.q1_target, self.q1, self.tau)
            soft_update(self.q2_target, self.q2, self.tau)


def main():
    env_name = 'Pendulum-v1'
    state_size = 3
    action_shape = (1)
    max_episode_steps = 200
    env = CustomEnv(gym.make(env_name), max_episode_steps)

    agent = Agent(state_size)

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

        for j in range(20):
            agent.update(j & 1)

        logger.log(evaluate_policy(agent, CustomEnv(gym.make(env_name), max_episode_steps), action_shape))

        if episode % dump_interval == 0:
            progress.set_postfix_str(f"avg_steps = {logger.average_score_log[-1]:.2f}")

    logger.plot()

if __name__ == '__main__':
    main()
