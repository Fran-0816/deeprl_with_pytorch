from typing import Tuple

import gymnasium as gym
import torch


class CustomEnv(gym.Wrapper):
    """学習環境のカスタムラッパー.

     - step 上限でのエピソード打ち切り
     - エピソードのスコア保持
     - 返り値を torch.Tensor に変換
    を行う.
    """
    def __init__(self, env: gym.Env, max_episode_steps: int) -> None:
        super().__init__(env)

        self._max_episode_steps = max_episode_steps
        self._elapesed_steps = None

        self._score = None

    def step(self, action: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        next_state, reward, done, truncated, info = self.env.step(action)
        self._elapesed_steps += 1

        self._score += reward

        if self._elapesed_steps >= self._max_episode_steps:
            done = True

        return (
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([done], dtype=torch.long)
        )

    def reset(self) -> torch.Tensor:
        initial_state, _ = self.env.reset()
        self._elapesed_steps = 0
        self._score = 0.0
        return torch.tensor(initial_state, dtype=torch.float32)

    @property
    def score(self) -> float:
        return self._score
