"""Actor-Critic (モンテカルロ) の実装"""

from tqdm.auto import trange
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim

from network import SoftmaxPolicyNet, VFunctionNet
from env import CustomEnv
from memory import TransitionMemory
from util import ScoreLogger


class AgentConfig:
    lr_policy = 2e-3
    lr_v = 3e-3
    gamma = 0.98


class Agent(AgentConfig):
    def __init__(self, state_size, action_size):
        self.policy = SoftmaxPolicyNet(state_size, action_size)
        self.v = VFunctionNet(state_size)

        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': self.lr_policy},
            {'params': self.v.parameters(), 'lr': self.lr_v}
        ])

        self.trajectory = TransitionMemory()

    def get_action(self, state, explore=False):
        with torch.no_grad():
            pi = self.policy(state)
            if not explore:
                return pi.argmax(dim=-1, keepdim=True)
            action = pi.multinomial(1)
            return action

    def update(self):
        states, actions, rewards, _, _ = self.trajectory.get_batch()

        returns = []
        G = torch.tensor(0.0)
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns = torch.stack(returns[::-1])

        vs = self.v(states)
        loss_v = F.mse_loss(vs, returns)

        advantages = (returns - vs).detach()

        pis = self.policy(states)
        probs = pis.gather(dim=1, index=actions)
        loss_policy = (-advantages * torch.log(probs)).mean()

        self.optimizer.zero_grad()
        loss_policy.backward()
        loss_v.backward()
        self.optimizer.step()


def main():
    env_name = 'CartPole-v1'
    state_size = 4
    action_size = 2
    action_shape = ()
    max_episode_steps = 500
    env = CustomEnv(gym.make(env_name), max_episode_steps)
    agent = Agent(state_size, action_size)

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
            agent.trajectory.add_transition(state, action, reward, next_state, done)
            state = next_state

        agent.update()

        logger.log(env.score)

        if episode % dump_interval == 0:
            progress.set_postfix_str(f"avg_steps = {logger.average_score_log[-1]:.2f}")

    logger.plot()

if __name__ == '__main__':
    main()
