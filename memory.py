import random
from collections import deque
from typing import Tuple
import numpy as np
import torch


class TransitionMemory:
    """方策オン型アルゴリズム用の遷移メモリ.

    Note:
        一時的に遷移を格納するために DQN のマルチステップ学習でも用いる.
    """
    def __init__(self) -> None:
        self.memory = []

    def add_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor
    ) -> None:
        transition = (state, action, reward, next_state, done)
        self.memory.append(transition)

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards, next_states, dones = zip(*self.memory)
        self.memory = []
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(dones)
        )

    def __len__(self) -> int:
        return len(self.memory)


class ReplayMemory:
    """経験再生メモリ."""
    def __init__(self, memory_size: int, batch_size: int) -> None:
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def add_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor
    ) -> None:
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mini_batch = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(dones)
        )

    def __len__(self) -> int:
        return len(self.memory)


class MultiStepReplayMemory:
    """multistep learning 用の経験再生メモリ."""
    def __init__(self, memory_size: int, batch_size : int) -> None:
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def add_experience(self, state: torch.Tensor, action: torch.Tensor, td_target: torch.Tensor) -> None:
        experience = (state, action, td_target)
        self.memory.append(experience)

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mini_batch = random.sample(self.memory, k=self.batch_size)
        state, action, td_target = zip(*mini_batch)

        return (
            torch.stack(state),
            torch.stack(action),
            torch.stack(td_target)
        )

    def __len__(self) -> int:
        return len(self.memory)


class PriorizedReplayMemory:
    """優先度付き経験再生用メモリ.

    SumTree は非使用. 優先度から算出した確率に基づいて np.random.choice で非復元抽出する.

    Attributes:
        len (int): 現在格納されている経験の数.
        head (int): 次に遷移を格納するインデックス.
    """
    def __init__(
        self,
        memory_size: int,
        batch_size: int,
        eps: float | np.ndarray | torch.Tensor = np.finfo(np.float32).eps
    ) -> None:
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.len = 0

        self.memory = [None] * memory_size
        self.head = 0
        self.priorities = torch.zeros(memory_size)

        self.alpha = 0.6
        self.beta = 0.4

        self.max_priority = torch.tensor(1.0)

        self.eps = eps

    def add_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor
    ) -> None:
        experience = (state, action, reward, next_state, done)
        self.memory[self.head] = experience
        self.priorities[self.head] = self.max_priority
        self.head = (self.head + 1) % self.memory_size
        self.len = min(self.memory_size, self.len + 1)

    def get_batch(self) -> Tuple[np.ndarray, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        priorities = (self.priorities[:self.len] + self.eps)**(self.alpha)
        probs = priorities / priorities.sum()
        indices = np.random.choice(np.arange(self.len), self.batch_size, replace=False, p=probs.numpy())

        self.beta = min(1, self.beta + 0.0003)

        weights = (probs[indices] * self.len)**(-self.beta)
        weights /= weights.max()
        weights.unsqueeze_(dim=-1)

        states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in indices])

        return (
            indices,
            weights,
            (torch.stack(states),
             torch.stack(actions),
             torch.stack(rewards),
             torch.stack(next_states),
             torch.stack(dones))
        )

    def update_priorities(self, indices: np.ndarray, priorities: torch.Tensor) -> None:
        self.priorities[indices] = priorities

        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self) -> int:
        return self.len
