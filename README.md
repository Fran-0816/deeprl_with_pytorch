# PyTorch で実装する深層強化学習
PyTorch による深層強化学習アルゴリズムの実装例

環境定義には [OpenAI Gym](https://www.gymlibrary.dev/) の後継である [Gymnasium](https://gymnasium.farama.org/) を使用.  
分散学習には [Ray](https://www.ray.io/) を使用.  
離散値制御は `CartPole-v1` 環境, 連続値制御は `Pendulum-v1` 環境で実施する.

## 動作要件
```
Python 3.10.6
```

## モジュール
1. [`env`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/env.py) : すべてのアルゴリズムで使う環境の定義
2. [`memory`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/memory.py) : 学習データ格納用のメモリ定義
3. [`network`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/network.py) : ニューラルネットワーク定義
4. [`util`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/util.py) : ユーティリティ

## アルゴリズム
1. PPO 編
   1. [`reinforce`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/01.reinforce.py) : REINFORCE
   2. [`actorcritic`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/02.actorcritic.py) : モンテカルロ法の Actor-Critic
   3. [`actorcritic_onestep`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/03.actorcritic_onestep.py) : TD 法の Actor-Critic
   4. [`gae`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/04.gae.py) : GAE
   5. [`actorcritic_fixedsteps`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/05.actorcritic_fixedsteps.py) : 固定ステップ経過ごとにパラメータ更新
   6. [`ppo`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/06.ppo.py) : PPO
2. DQN 編
   1. [`dqn`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/07.dqn.py) : DQN
   2. [`double_dqn`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/08.double_dqn.py) : Double DQN
   3. [`priorized_experience_replay`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/09.priorized_experience_replay.py) : 優先度付き経験再生 (SumTree 非使用)
   4. [`dueling_network`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/10.dueling_network.py) : Dueling Network
   5. [`multistep_learning`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/11.multistep_learning.py) : マルチステップ学習
   6. [`categorical_dqn`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/12.categorical_dqn.py) : Categorical DQN (C51)
   7. [`noisy_net`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/13.noisy_net.py) : Noisy Net
3. 分散学習編
   1. [`actorcritic_common`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/14.actorcritic_common.py) : Actor と Critic でパラメータを共有する
   2. [`entropy_reguralization`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/15.entropy_reguralization.py) : 方策エントロピー正則化
   3. [`a3c`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/16.a3c.py) : A3C
   4. [`a2c`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/17.a2c.py) : A2C
   5. [`ppo2`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/18.ppo2.py) : A2C の同期分散学習を PPO に適用
4. 連続値制御編
   1. [`ppo_continuous`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/19.ppo_continuous.py) : ガウス方策を用いた PPO
   2. [`ddpg`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/20.ddpg.py) : DDPG
   3. [`td3`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/21.td3.py) : TD3
   4. [`sac`](https://github.com/Fran-0816/deeprl_with_pytorch/blob/main/22.sac.py) : SAC (連続値制御)
5. And more ?
