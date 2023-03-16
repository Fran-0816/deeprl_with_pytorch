"""PPO で連続値制御"""

from tqdm.auto import trange
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from env import CustomEnv
from memory import TransitionMemory
from network import GaussianPolicyNet, VFunctionNet
from util import ScoreLogger, evaluate_policy, action_converter_for_pendulum


class AgentConfig:
    lr_policy = 3e-3
    lr_v = 3e-3
    gamma = 0.9
    gae_lambda = 0.9
    eps_clip = 0.2
    k_epoch = 15


class Agent(AgentConfig):
    def __init__(self, state_size, action_converter):
        self.policy = GaussianPolicyNet(state_size)
        self.v = VFunctionNet(state_size)

        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': self.lr_policy},
            {'params': self.v.parameters(), 'lr': self.lr_v}
        ])

        self.trajectory = TransitionMemory()

        self.action_converter = action_converter

    def get_action(self, state, convert_action=True, explore=False):
        action = None
        with torch.no_grad():
            mu, sigma = self.policy(state)
            if not explore:
                action = mu
            else:
                m = Normal(mu, sigma)
                action = m.sample()

        if convert_action:
            action = self.action_converter(action)

        return action

    def update(self):
        states, pre_squash_actions, rewards, next_states, dones = self.trajectory.get_batch()

        td_targets = rewards + (1 - dones) * self.gamma * self.v(next_states).detach()
        old_vs = self.v(states)

        deltas = (td_targets - old_vs).detach()
        advantages = []
        adv = torch.tensor(0.0)
        for d in reversed(deltas):
            adv = d + self.gamma * self.gae_lambda * adv
            advantages.append(adv)
        advantages = torch.stack(advantages[::-1])

        old_mus, old_sigmas = self.policy(states)
        old_pis = Normal(old_mus, old_sigmas)
        old_log_probs = old_pis.log_prob(pre_squash_actions).detach()

        for _ in range(self.k_epoch):
            vs = self.v(states)
            loss_v = F.mse_loss(vs, td_targets)

            mus, sigmas = self.policy(states)
            pis = Normal(mus, sigmas)
            log_probs = pis.log_prob(pre_squash_actions)
            ratios = torch.exp(log_probs - old_log_probs)

            surr1s = ratios * advantages
            surr2s = torch.clip(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_policy = -torch.min(surr1s, surr2s).mean()

            self.optimizer.zero_grad()
            loss_policy.backward()
            loss_v.backward()
            self.optimizer.step()


def main():
    env_name = 'Pendulum-v1'
    state_size = 3
    action_shape = (1)
    max_episode_steps = 200
    env = CustomEnv(gym.make(env_name), max_episode_steps)
    agent = Agent(state_size, action_converter_for_pendulum)

    num_updates = 1000
    update_steps = 100
    dump_interval = 50

    logger = ScoreLogger(dump_interval)

    progress = trange(1, num_updates + 1, dynamic_ncols=True)
    state = env.reset()
    done = False
    for update_t in progress:
        for _ in range(update_steps):
            pre_squash_action = agent.get_action(state, convert_action=False, explore=True)
            action = agent.action_converter(pre_squash_action)
            next_state, reward, done = env.step(action.numpy().reshape(action_shape))
            agent.trajectory.add_transition(state, pre_squash_action, reward, next_state, done)
            state = env.reset() if done else next_state

        agent.update()

        logger.log(evaluate_policy(agent, CustomEnv(gym.make(env_name), max_episode_steps), action_shape))

        if update_t % dump_interval == 0:
            progress.set_postfix_str(f"avg_steps = {logger.average_score_log[-1]:.2f}")

    logger.plot()

if __name__ == '__main__':
    main()
