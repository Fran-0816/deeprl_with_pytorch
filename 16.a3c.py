"""A3C の実装"""

from tqdm.auto import trange
import ray
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim

from env import CustomEnv
from memory import TransitionMemory
from network import ActorCriticNet
from util import ScoreLogger, evaluate_policy


class WorkerConfig:
    gamma = 0.98
    gae_lambda = 0.95
    v_coef = 1
    entropy_coef = 0


@ray.remote
class Worker(WorkerConfig):
    def __init__(self, worker_id, env, state_size, action_size, action_shape):
        self.worker_id = worker_id

        self.local_env = env
        self.local_model = ActorCriticNet(state_size, action_size)

        self.state = self.local_env.reset()
        self.done = False

        self.action_shape = action_shape

    def generate_grad(self, global_model, num_steps, sync_param):
        if sync_param:
            self.local_model.load_state_dict(global_model.state_dict())

        trajectory = TransitionMemory()
        for _ in range(num_steps):
            action = self.get_action(self.state)
            next_state, reward, self.done = self.local_env.step(action.numpy().reshape(self.action_shape))
            trajectory.add_transition(self.state, action, reward, next_state, self.done)
            self.state = self.local_env.reset() if self.done else next_state

        return self.worker_id, self.compute_grad(trajectory)

    def get_action(self, state):
        with torch.no_grad():
            pi = self.local_model.pi(state)
            action = pi.multinomial(1)
            return action

    def compute_grad(self, trajectory):
        states, actions, rewards, next_states, dones = trajectory.get_batch()

        td_targets = rewards + (1 - dones) * self.gamma * self.local_model.v(next_states).detach()
        vs = self.local_model.v(states)
        loss_v = F.mse_loss(vs, td_targets)

        deltas = (td_targets - vs).detach()
        advantages = []
        adv = torch.tensor(0.0)
        for d in reversed(deltas):
            adv = d + self.gamma * self.gae_lambda * adv
            advantages.append(adv)
        advantages = torch.stack(advantages[::-1]).detach()

        pis = self.local_model.pi(states)
        log_pis = torch.log(pis)
        log_probs = log_pis.gather(dim=1, index=actions)
        loss_policy = (-advantages * log_probs).mean()

        entropy = -(pis * log_pis).sum(dim=-1).mean()

        loss = loss_policy + self.v_coef * loss_v - self.entropy_coef * entropy

        self.local_model.zero_grad()
        loss.backward()

        grads = []

        for param in self.local_model.parameters():
            grads.append(param.grad)

        return grads


class HostConfig:
    lr_common = 5e-4
    lr_policy = 5e-4
    lr_v = 3e-3


class Host(HostConfig):
    def __init__(self, state_size, action_size):
        self.global_model = ActorCriticNet(state_size, action_size)

        self.optimizer = optim.Adam([
            {'params': self.global_model.common.parameters(), 'lr': self.lr_common},
            {'params': self.global_model.actor.parameters(), 'lr': self.lr_policy},
            {'params': self.global_model.critic.parameters(), 'lr': self.lr_v}
        ])

    def get_action(self, state):
        with torch.no_grad():
            pi = self.global_model.pi(state)
            action = pi.argmax(dim=-1, keepdim=True)
            return action

    def update(self, grads):
        self.optimizer.zero_grad()
        for global_param, grad in zip(self.global_model.parameters(), grads):
            global_param._grad = grad
        self.optimizer.step()


def main():
    ray.init()

    env_name = 'CartPole-v1'
    state_size = 4
    action_size = 2
    action_shape = ()
    max_episode_steps = 500

    num_worker = 3
    num_update = 3000
    update_steps = 100
    sync_interval = 5
    dump_interval = 50

    host_agent = Host(state_size, action_size)

    logger = ScoreLogger(dump_interval)

    workers = [Worker.remote(i, CustomEnv(gym.make(env_name), max_episode_steps), state_size, action_size, action_shape) for i in range(num_worker)]
    grad_send_counter = [0] * num_worker
    progress_jobs = [worker.generate_grad.remote(host_agent.global_model, update_steps, sync_param=True) for worker in workers]

    progress = trange(1, num_update + 1, dynamic_ncols=True)
    for update_t in progress:
        finish_job, progress_jobs = ray.wait(progress_jobs, num_returns=1)
        worker_id, grads = ray.get(finish_job)[0]
        grad_send_counter[worker_id] += 1

        host_agent.update(grads)

        sync_param = (grad_send_counter[worker_id] % sync_interval) == 0
        progress_jobs.extend([workers[worker_id].generate_grad.remote(host_agent.global_model, update_steps, sync_param)])

        logger.log(evaluate_policy(host_agent, CustomEnv(gym.make(env_name), max_episode_steps), action_shape))

        if update_t % dump_interval == 0:
            progress.set_postfix_str(f"avg_steps = {logger.average_score_log[-1]:.2f}")

    logger.plot()

    ray.shutdown()

if __name__ == '__main__':
    main()