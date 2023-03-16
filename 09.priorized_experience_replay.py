"""優先度付き経験再生の適用"""

import random
from tqdm.auto import trange
import gymnasium as gym
import torch
import torch.optim as optim

from env import CustomEnv
from memory import PriorizedReplayMemory
from network import DQN
from util import ScoreLogger, evaluate_policy, hard_update


class AgentConfig:
    lr = 1e-4
    epsilon = 0.3
    gamma = 0.98
    memory_size = 10000
    batch_size = 32
    min_num_experience = 2000


class Agent(AgentConfig):
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.q = DQN(state_size, action_size)
        self.q_target = DQN(state_size, action_size)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)

        self.experience = PriorizedReplayMemory(self.memory_size, self.batch_size)

    def get_action(self, state, explore=False):
        if explore and random.random() < self.epsilon:
            return torch.randint(0, self.action_size, size=(1,))
        else:
            q = self.q(state)
            action = q.argmax(dim=-1, keepdim=True)
            return action

    def update(self):
        if len(self.experience) < self.min_num_experience:
            return

        indices, weights, (states, actions, rewards, next_states, dones) = self.experience.get_batch()

        qs = self.q(states)
        qs = qs.gather(dim=1, index=actions)

        target_qs = self.q_target(next_states).detach()
        target_qs = target_qs.max(dim=1, keepdim=True)[0]
        td_targets = rewards + (1 - dones) * self.gamma * target_qs

        td_errors = td_targets - qs
        loss = ((td_errors**2) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        td_errors = td_errors.view(-1).detach()
        self.experience.update_priorities(indices, torch.abs(td_errors))


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
            agent.experience.add_experience(state, action, reward, next_state, done)
            state = next_state

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
