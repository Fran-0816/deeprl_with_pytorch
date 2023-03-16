"""PPO の実装"""

from tqdm.auto import trange
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim

from network import SoftmaxPolicyNet, VFunctionNet
from env import CustomEnv
from memory import TransitionMemory
from util import ScoreLogger, evaluate_policy


class AgentConfig:
    lr_policy = 1e-3
    lr_v = 3e-3
    gamma = 0.98
    gae_lambda = 0.95
    eps_clip = 0.1
    k_epoch = 3


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
        states, actions, rewards, next_states, dones = self.trajectory.get_batch()

        td_targets = rewards + (1 - dones) * self.gamma * self.v(next_states).detach()
        old_vs = self.v(states)

        deltas = (td_targets - old_vs).detach()
        advantages = []
        adv = torch.tensor(0.0)
        for d in reversed(deltas):
            adv = d + self.gamma * self.gae_lambda * adv
            advantages.append(adv)
        advantages = torch.stack(advantages[::-1])

        old_pis = self.policy(states).detach()
        old_probs = old_pis.gather(dim=1, index=actions)
        old_log_probs = torch.log(old_probs)

        for _ in range(self.k_epoch):
            vs = self.v(states)
            loss_v = F.mse_loss(vs, td_targets)

            pis = self.policy(states)
            probs = pis.gather(dim=1, index=actions)
            ratios = torch.exp(torch.log(probs) - old_log_probs)

            surr1s = ratios * advantages
            surr2s = torch.clip(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_policy = -torch.min(surr1s, surr2s).mean()

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

    num_updates = 1000
    update_steps = 100
    dump_interval = 50

    logger = ScoreLogger(dump_interval)

    progress = trange(1, num_updates + 1, dynamic_ncols=True)
    state = env.reset()
    done = False
    for update_t in progress:
        for _ in range(update_steps):
            action = agent.get_action(state, explore=True)
            next_state, reward, done = env.step(action.numpy().reshape(action_shape))
            agent.trajectory.add_transition(state, action, reward, next_state, done)
            state = env.reset() if done else next_state

        agent.update()

        logger.log(evaluate_policy(agent, CustomEnv(gym.make(env_name), max_episode_steps), action_shape))

        if update_t % dump_interval == 0:
            progress.set_postfix_str(f"avg_steps = {logger.average_score_log[-1]:.2f}")

    logger.plot()

if __name__ == '__main__':
    main()
