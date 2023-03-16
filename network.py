from typing import Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxPolicyNet(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        pi = F.softmax(self.fc3(x), dim=-1)

        return pi


class VFunctionNet(nn.Module):
    def __init__(self, state_size: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        v = self.fc3(x)

        return v


class ActorCriticNet(nn.Module):
    """Actor (Softmax 方策) と Critic (状態価値関数) のパラメータを共有.

    拡張性を上げるために, 共通部分, Actor 部分, Critic 部分をそれぞれ nn.ModuleDict で囲んだ.
    """
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()

        self.common = nn.ModuleDict({
            'fc1': nn.Linear(state_size, 64),
            'fc2': nn.Linear(64, 64)
        })
        self.actor = nn.ModuleDict({
            'fc1': nn.Linear(64, action_size)
        })
        self.critic = nn.ModuleDict({
            'fc1': nn.Linear(64, 1)
        })

    def pi(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.common['fc1'](x))
        x = F.relu(self.common['fc2'](x))

        pi = F.softmax(self.actor['fc1'](x), dim=-1)

        return pi

    def v(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.common['fc1'](x))
        x = F.relu(self.common['fc2'](x))

        v = self.critic['fc1'](x)

        return v


class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        q = self.fc3(x)

        return q


class DuelingDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()

        self.common = nn.ModuleDict({
            'fc1': nn.Linear(state_size, 64),
        })
        self.value = nn.ModuleDict({
            'fc1': nn.Linear(64, 64),
            'fc2': nn.Linear(64, 1)
        })
        self.advantage = nn.ModuleDict({
            'fc1': nn.Linear(64, 64),
            'fc2': nn.Linear(64, action_size)
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.common['fc1'](x))

        state_value = self.value['fc1'](x)
        state_value = self.value['fc2'](x)

        advantage = self.advantage['fc1'](x)
        advantage = self.advantage['fc2'](x)

        q = state_value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q


class NoisyLinear(nn.Module):
    """Noisy Net 用のノイズ付き全結合層.

    Factorised Gaussian noise で実装.
    """
    def __init__(self, input_features: int, output_features: int, sigma: float = 0.5) -> None:
        super().__init__()

        self.sigma = sigma
        self.input_features = input_features
        self.output_features = output_features

        self.mu_weight = nn.Parameter(torch.FloatTensor(output_features, input_features))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(output_features, input_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(output_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(output_features))

        self.register_buffer('epsilon_input', torch.FloatTensor(input_features))
        self.register_buffer('epsilon_output', torch.FloatTensor(output_features))

        self.bound = input_features ** -0.5

        self.reset_parameters()

        self.sample_noise()

    def forward(self, x: torch.Tensor, sample_noise: bool = True) -> torch.Tensor:
        if not self.training:
            return F.linear(x, weight=self.mu_weight, bias=self.mu_bias)

        if sample_noise:
            self.sample_noise()

        return F.linear(x, weight=self.mu_weight, bias=self.bias)

    @property
    def weight(self) -> torch.Tensor:
        return self.sigma_weight * torch.outer(self.epsilon_output, self.epsilon_input) + self.mu_weight

    @property
    def bias(self) -> torch.Tensor:
        return self.sigma_bias * self.epsilon_output + self.mu_bias

    def sample_noise(self) -> None:
        self.epsilon_input = self.get_noise_tensor(self.input_features)
        self.epsilon_output = self.get_noise_tensor(self.output_features)

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.mu_weight, -self.bound, self.bound)
        nn.init.constant_(self.sigma_weight, self.sigma * self.bound)
        nn.init.uniform_(self.mu_bias, -self.bound, self.bound)
        nn.init.constant_(self.sigma_bias, self.sigma * self.bound)

    def get_noise_tensor(self, features: int) -> torch.Tensor:
        noise = torch.randn(features)
        return torch.sign(noise) * torch.sqrt(torch.abs(noise))


class NoisyDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()

        self.fc1 = NoisyLinear(state_size, 64)
        self.fc2 = NoisyLinear(64, 64)
        self.fc3 = NoisyLinear(64, action_size)

    def forward(self, x: torch.Tensor, sample_noise: bool = True) -> torch.Tensor:
        x = F.relu(self.fc1(x, sample_noise))
        x = F.relu(self.fc2(x, sample_noise))

        q = self.fc3(x, sample_noise)

        return q


class CategoricalDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, num_atoms: int = 51) -> None:
        super().__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size * num_atoms)

        self.action_size = action_size
        self.num_atoms = num_atoms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        logit = x.view(-1, self.action_size, self.num_atoms)

        dist = F.softmax(logit, dim=-1)

        return dist


class GaussianPolicyNet(nn.Module):
    def __init__(self, state_size: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu_fc1 = nn.Linear(64, 1)
        self.log_sigma_fc1 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.mu_fc1(x)

        log_sigma = self.log_sigma_fc1(x)
        # sigma が大きくならないように clip.
        log_sigma = torch.clip(log_sigma, -5, 2)
        sigma = torch.exp(log_sigma)

        return mu, sigma


class DeterministicPolicyNet(nn.Module):
    """決定論的方策.

    fc3 の出力 x は (-inf, inf) の範囲をとるので, action_converter に通して環境に合った出力に変換する.
    例えば Pendulum-v1 なら 行動空間が [-2, 2] だから, tanh(x) * 2 で変換すればいい.
    """
    def __init__(self, state_size: int, action_converter: Callable) -> None:
        super().__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

        self.converter = action_converter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        action = self.converter(x)

        return action


class QFunctionNet(nn.Module):
    def __init__(self, state_size: int) -> None:
        super().__init__()

        self.state_fc1 = nn.Linear(state_size, 32)
        self.action_fc1 = nn.Linear(1, 32)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.state_fc1(state))
        h2 = F.relu(self.action_fc1(action))
        cat = torch.cat((h1, h2), dim=-1)

        q = F.relu(self.fc1(cat))
        q = self.fc2(q)

        return q
