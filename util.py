from typing import List, Tuple
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from env import CustomEnv


class ScoreLogger:
    """各エピソードのスコア記録用の logger

    各エピソードのスコアと, 末尾 span エピソード分の平均スコアを管理する.
    """
    def __init__(self, span: int) -> None:
        self.score_log = []
        self.average_score_log = []

        self.span = span
        self.span_log = deque()
        self.span_sum = 0

    def log(self, step):
        self.score_log.append(step)

        if len(self.span_log) == self.span:
            self.span_sum -= self.span_log[0]
            self.span_log.popleft()
        self.span_sum += step
        self.span_log.append(step)

        self.average_score_log.append(self.span_sum / len(self.span_log))

    def plot(self):
        # 各エピソードのスコアと, 平均を同じグラフに出力.
        plt.plot(self.score_log)
        plt.plot(self.average_score_log)
        plt.show()


def evaluate_policy(agent, test_env: CustomEnv, action_shape: Tuple[()] | Tuple[int, ...], T: int = 1):
    # agent が持つ方策の性能を測定する.
    average_score = 0.0
    for _ in range(T):
        state = test_env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            state, _, done = test_env.step(action.numpy().reshape(action_shape))

        average_score += test_env.score

    return average_score / T


def flatten_trajectories(trajectories: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]):
    """Trajectory の平滑化

    同期型分散学習で各エージェントが集めた複数の trajectory を 1 次元のリストに集約する.
    """
    state, action, reward, next_state, done = zip(*trajectories)
    return torch.cat(state), torch.cat(action), torch.cat(reward), torch.cat(next_state), torch.cat(done)


def hard_update(model: nn.Module, target_model: nn.Module):
    model.load_state_dict(target_model.state_dict())


def soft_update(model: nn.Module, target_model: nn.Module, tau: float):
    for param, param_target in zip(model.parameters(), target_model.parameters()):
        param.data.copy_(param.data * (1 - tau) + param_target.data * tau)


def action_converter_for_pendulum(x: torch.Tensor) -> torch.Tensor:
    """Pendulum-v1 用の行動変換器

    以下の流れでネットワークの出力 in (-inf, inf) を Pendulum-v1 の行動空間 [-2, 2] に変換.
    (-inf, inf) -- tanh -> [-1, 1] -- x2 -> [-2, 2].
    """
    return torch.tanh(x) * 2


class OUNoise:
    """Ornstein-Uhlenbeck noise 生成器.

    エピソードごとに reset を呼び, ノイズの振れ幅を小さくしていく.
    """
    def __init__(self, mu=0.0, theta=0.1, sigma=0.5, sigma_min = 0.05, sigma_decay=0.997):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay

    def reset(self):
        self.x = torch.tensor([self.mu])
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

    def sample(self):
        dx = self.theta * (self.mu - self.x) + self.sigma * torch.randn(1)
        self.x += dx
        return self.x


def log_prob_of_squashed_action(
    pre_squash_dist: torch.distributions.Distribution,
    pre_squash_action: torch.Tensor,
    eps: float | np.ndarray | torch.Tensor = np.finfo(np.float32).eps
) -> torch.Tensor:
    # tanh で変換する前の行動選択確率の対数を求める.
    return -(pre_squash_dist.log_prob(pre_squash_action) - torch.log(1 - torch.tanh(pre_squash_action)**2 + eps))
