"""DDPG の実装"""

from tqdm.auto import trange
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim

from env import CustomEnv
from memory import ReplayMemory
from network import DeterministicPolicyNet, QFunctionNet
from util import ScoreLogger, evaluate_policy, soft_update, action_converter_for_pendulum, OUNoise


class AgentConfig:
    lr_policy = 2e-3
    lr_q = 1e-2
    gamma = 0.98
    memory_size = 10000
    batch_size = 32
    min_num_experience = 2000
    tau = 0.005


class Agent(AgentConfig):
    def __init__(self, state_size):
        self.policy = DeterministicPolicyNet(state_size, action_converter_for_pendulum)
        self.policy_target = DeterministicPolicyNet(state_size, action_converter_for_pendulum)
        self.policy_target.load_state_dict(self.policy.state_dict())

        self.q = QFunctionNet(state_size)
        self.q_target = QFunctionNet(state_size)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer_policy = optim.Adam(self.policy.parameters(), self.lr_policy)
        self.optimizer_q = optim.Adam(self.q.parameters(), self.lr_q)

        self.experience = ReplayMemory(self.memory_size, self.batch_size)

        self.noise = OUNoise()

    def get_action(self, state, explore=False):
        with torch.no_grad():
            action = self.policy(state)
            if explore:
                action = action + self.noise.sample()
                action = torch.clip(action, -2, 2)

            return action

    def update(self):
        if len(self.experience) < self.min_num_experience:
            return

        states, actions, rewards, next_states, dones = self.experience.get_batch()

        qs = self.q(states, actions)
        td_targets = rewards + self.gamma * (1 - dones) * self.q_target(next_states, self.policy_target(next_states)).detach()
        loss_q = F.mse_loss(qs, td_targets)

        self.optimizer_q.zero_grad()
        loss_q.backward()
        self.optimizer_q.step()

        loss_policy = -self.q(states, self.policy(states)).mean()

        self.optimizer_policy.zero_grad()
        loss_policy.backward()
        self.optimizer_policy.step()

        soft_update(self.policy_target, self.policy, self.tau)
        soft_update(self.q_target, self.q, self.tau)


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
        agent.noise.reset()
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state, explore=True)
            next_state, reward, done = env.step(action.numpy().reshape(action_shape))
            agent.experience.add_experience(state, action, reward, next_state, done)
            state = next_state

        for _ in range(20):
            agent.update()

        logger.log(evaluate_policy(agent, CustomEnv(gym.make(env_name), max_episode_steps), action_shape))

        if episode % dump_interval == 0:
            progress.set_postfix_str(f"avg_score = {logger.average_score_log[-1]:.2f}")

    logger.plot()

if __name__ == '__main__':
    main()
