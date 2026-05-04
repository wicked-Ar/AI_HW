# Homework #1 AI Coding Tool Guide

이 문서는 Codex, Claude Code, GitHub Copilot Agent 등 AI coding tool에게 제공하기 위한 작업 지시서이다.  
목표는 `linklab/link_rl` 수업 코드를 기반으로 Homework #1을 수행하기 위한 코드 수정, 실행, 결과 캡처, 보고서 작성 준비를 돕는 것이다.

---

## 0. 절대 조건

다음 조건은 과제 요구사항이므로 임의로 변경하지 않는다.

### Problem 1: Slippery FrozenLake

- 기존 수업 코드 기반:
  - `link_rl/_01_code/_08_SARSA_Q_Learning/a_frozen_lake_q_learning.py`
- 환경:
  - `FrozenLake-v1`
  - `map_name="4x4"`
  - `is_slippery=True`
  - `success_rate=1.0/3.0` 개념에 해당하는 stochastic FrozenLake
- 고정 config:

```python
config = {
    "num_episodes": 50_000,
    "validation_time_steps_interval": 1_000,
    "validation_num_episodes": 10,
    "alpha": 0.1,
    "gamma": 0.95,
    "epsilon": 0.1,
}
```

- 보고서에 포함해야 할 것:
  - 학습 및 검증 완료 콘솔 캡처
  - 최종 play 성공 콘솔 캡처
  - Goal에 도착한 FrozenLake 화면 캡처
  - WandB 그래프 4개 캡처
  - 각 WandB 그래프 4개에 대한 해석

### Problem 2: LunarLander

- 환경:
  - `gymnasium.make("LunarLander-v3")`
- 알고리즘:
  - DQN
- 제출해야 할 구현 코드 파일:
  - `qnet_lunar_lander.py`
  - `dqn_train_test.py`
  - `dqn_lunar_lander_train.py`
  - `dqn_lunar_lander_test.py`
- 고정 config 항목:
  - `episode_reward_avg_solved = 200`
  - `validation_num_episodes = 3`
- 비교 실험:
  - `learning_rate = 0.001`
  - `learning_rate = 0.00001`
- 보고서에 포함해야 할 WandB 지표:
  - `[VALIDATION] Mean Episode Reward`
  - `[TRAIN] Episode Reward`
  - `[TRAIN] Loss`
  - `[TRAIN] Epsilon`
  - `[TRAIN] Replay buffer`
- 보고서에 포함해야 할 것:
  - `dqn_lunar_lander_test.py` 실행 성공 콘솔 캡처
  - 성공한 LunarLander 화면 또는 영상 캡처
  - Learning rate 2개 비교 그래프
  - 5개 그래프 각각의 해석
  - 더 좋은 learning rate 선택 및 이유

---

## 1. 현재 GitHub 코드 기준 작업 대상

Repository:

```text
https://github.com/linklab/link_rl
```

주요 폴더:

```text
link_rl/_01_code/_08_SARSA_Q_Learning/
link_rl/_01_code/_09_DQN/
```

기존 FrozenLake Q-learning 코드:

```text
_01_code/_08_SARSA_Q_Learning/a_frozen_lake_q_learning.py
```

기존 DQN CartPole 코드:

```text
_01_code/_09_DQN/c_qnet.py
_01_code/_09_DQN/d_dqn_train_test.py
_01_code/_09_DQN/e_dqn_cartpole_train.py
_01_code/_09_DQN/f_dqn_cartpole_test.py
```

Problem 2는 기존 CartPole DQN 코드를 LunarLander용으로 복사·수정하는 방식으로 수행한다.

---

## 2. GitHub pull 및 초기화

`link_rl` 폴더에서 다음 명령을 실행한다.

```bash
git checkout -- _01_code
git pull
```

주의:

- 기존에 수정한 코드가 있다면 `git checkout -- _01_code` 명령으로 사라질 수 있다.
- 이미 작업한 코드가 있다면 먼저 백업한다.

---

## 3. Python 환경 준비

권장 Python 버전:

```text
Python >= 3.10
```

필요 패키지 예시:

```bash
pip install numpy torch gymnasium wandb pygame moviepy
pip install "gymnasium[box2d]"
```

LunarLander는 Box2D 의존성이 필요하므로 `gymnasium[box2d]` 설치가 필요할 수 있다.

WandB 사용 시:

```bash
wandb login
```

---

## 4. Problem 1: Slippery FrozenLake 작업 지시

### 4.1 목표

`a_frozen_lake_q_learning.py`를 실행하여 stochastic FrozenLake 환경에서 Q-learning을 수행한다.  
최종적으로 validation 평균 reward가 충분히 높아져 `TRAINING DONE`이 출력되고, 이후 play 단계에서 `PLAY EPISODE SUCCESS!!!`가 출력되는 화면을 확보한다.

### 4.2 확인 및 수정할 항목

`a_frozen_lake_q_learning.py`에서 다음 값이 설정되어 있는지 확인한다.

```python
IS_SLIPPERY = True
MAP_NAME = "4x4"
DESC = None
```

config는 다음과 같아야 한다.

```python
config = {
    "num_episodes": 50_000,
    "validation_time_steps_interval": 1_000,
    "validation_num_episodes": 10,
    "alpha": 0.1,
    "gamma": 0.95,
    "epsilon": 0.1,
}
```

위 config는 과제 고정 조건이므로 수정하지 않는다.

### 4.3 실행 명령

```bash
cd link_rl/_01_code/_08_SARSA_Q_Learning
python a_frozen_lake_q_learning.py
```

### 4.4 실패할 경우 확인할 점

Stochastic FrozenLake는 미끄러짐이 있는 환경이므로 학습이 완료되어도 play 1회가 실패할 수 있다.  
`PLAY EPISODE FAILED!!!`가 나오면 다시 실행하여 성공 화면을 확보한다.

확인할 로그:

```text
***** TRAINING DONE!!! *****
PLAY EPISODE SUCCESS!!!
```

### 4.5 WandB 그래프 해석 방향

보고서에는 다음 4개 그래프를 캡처하고 해석한다.

#### `[VALIDATION] Average Episode Reward`

- 검증 episode에서 평균적으로 Goal에 도달한 비율을 의미한다.
- FrozenLake는 Goal 도달 reward가 1이고 실패 reward가 0에 가까우므로, 평균 reward가 1에 가까울수록 성공률이 높다.
- Stochastic 환경에서는 이동 결과가 확률적으로 달라지므로 그래프가 완전히 매끄럽게 증가하지 않고 진동할 수 있다.

#### `[TRAIN] Length of Visited States`

- 한 episode 동안 방문한 state 개수를 의미한다.
- 너무 짧은 경우 Hole에 빠져 조기 종료되었을 가능성이 있다.
- 학습이 진행되면 더 많은 state를 거쳐 Goal까지 도달하거나, 안정적인 경로를 찾는 경향이 나타날 수 있다.
- 단, slippery 환경에서는 같은 정책이라도 실제 이동 경로가 매번 달라져 변동성이 남는다.

#### `[TRAIN] Episode Reward`

- 각 학습 episode에서 얻은 reward를 의미한다.
- FrozenLake에서는 Goal에 도달하면 1, 실패하면 0이므로 성공 episode 발생 여부에 가깝게 해석할 수 있다.
- 학습 초기에는 대부분 0이지만, Q-table이 갱신되면서 1이 나타나는 빈도가 증가한다.

#### `[TRAIN] Average Episode TD Error`

- Q-learning 업데이트 과정에서 발생한 TD error의 평균값이다.
- 학습 초기에는 Q-value가 불안정하므로 TD error 변동이 클 수 있다.
- 학습이 진행되면 Q-table이 수렴하면서 TD error의 변동 폭이 감소하는 경향이 있다.
- 다만 stochastic 환경에서는 전이 결과가 확률적이므로 TD error가 완전히 0으로 고정되지는 않을 수 있다.

---

## 5. Problem 2: LunarLander DQN 작업 지시

### 5.1 목표

기존 `_09_DQN`의 CartPole DQN 코드를 복사하여 LunarLander-v3용 DQN 코드 4개를 작성한다.

제출 파일명:

```text
qnet_lunar_lander.py
dqn_train_test.py
dqn_lunar_lander_train.py
dqn_lunar_lander_test.py
```

### 5.2 파일 복사

```bash
cd link_rl/_01_code/_09_DQN
cp c_qnet.py qnet_lunar_lander.py
cp d_dqn_train_test.py dqn_train_test.py
cp e_dqn_cartpole_train.py dqn_lunar_lander_train.py
cp f_dqn_cartpole_test.py dqn_lunar_lander_test.py
```

### 5.3 `qnet_lunar_lander.py` 수정 방향

기존 QNet 구조는 그대로 사용해도 된다.

핵심은 LunarLander에서 사용할 때 다음처럼 입력과 출력 차원을 지정하는 것이다.

```python
qnet = QNet(n_features=8, n_actions=4)
target_qnet = QNet(n_features=8, n_actions=4)
```

이유:

- LunarLander state는 8차원이다.
  - x position
  - y position
  - x velocity
  - y velocity
  - angle
  - angular velocity
  - left leg contact
  - right leg contact
- LunarLander action은 4개이다.
  - do nothing
  - fire left engine
  - fire main engine
  - fire right engine

### 5.4 `dqn_train_test.py` 수정 방향

기본 구조는 CartPole DQN 코드와 동일하게 유지한다.

유지해야 할 구성:

- `ReplayBuffer`
- `Transition`
- `DqnTrainer`
- `DqnTester`
- epsilon-greedy action selection
- target network synchronization
- validation loop
- model save/load
- WandB logging

수정이 필요할 수 있는 부분:

1. 주석의 CartPole 설명을 LunarLander로 변경한다.
2. 파일명 또는 import가 CartPole에 종속되어 있으면 제거한다.
3. `DqnTester`에서 model load 파일명이 `dqn_{env_name}_latest.pth` 형태인지 확인한다.
4. `DqnTester.test()` 출력 오타가 있다면 보고서 캡처를 위해 다음처럼 수정해도 된다.

```python
print("[TOTAL_STEPS: {0:3d}, EPISODE REWARD: {1:4.1f}]".format(time_steps, episode_reward))
```

### 5.5 `dqn_lunar_lander_train.py` 작성 방향

기존 `e_dqn_cartpole_train.py`를 기반으로 다음 내용을 반영한다.

```python
import gymnasium as gym
import torch
import os

from qnet_lunar_lander import QNet
from dqn_train_test import DqnTrainer

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    print("TORCH VERSION:", torch.__version__)

    ENV_NAME = "LunarLander-v3"

    env = gym.make(ENV_NAME)
    valid_env = gym.make(ENV_NAME)

    config = {
        "env_name": ENV_NAME,
        "max_num_episodes": 3000,
        "batch_size": 64,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "steps_between_train": 1,
        "replay_buffer_size": 1_000_000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_final_scheduled_percent": 0.75,
        "print_episode_interval": 10,
        "target_sync_time_steps_interval": 500,
        "validation_time_steps_interval": 10_000,
        "validation_num_episodes": 3,
        "episode_reward_avg_solved": 200,
    }

    qnet = QNet(n_features=8, n_actions=4)
    target_qnet = QNet(n_features=8, n_actions=4)

    use_wandb = True

    dqn = DqnTrainer(
        env=env,
        valid_env=valid_env,
        qnet=qnet,
        target_qnet=target_qnet,
        config=config,
        use_wandb=use_wandb,
        current_dir=CURRENT_DIR,
    )

    dqn.train_loop()


if __name__ == "__main__":
    main()
```

주의:

- 과제에서 고정한 `validation_num_episodes=3`, `episode_reward_avg_solved=200`은 변경하지 않는다.
- learning rate 비교를 위해 `learning_rate=0.001`과 `learning_rate=0.00001` 실험을 각각 수행한다.
- 다른 하이퍼파라미터는 학습 성공을 위해 조정할 수 있지만, 보고서에는 변경한 값을 명시한다.
- 단, 교수자가 명시적으로 고정한 값은 절대 수정하지 않는다.

### 5.6 `dqn_lunar_lander_test.py` 작성 방향

기존 `f_dqn_cartpole_test.py`를 기반으로 다음 내용을 반영한다.

```python
import gymnasium as gym
import os

from qnet_lunar_lander import QNet
from dqn_train_test import DqnTester

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    ENV_NAME = "LunarLander-v3"

    test_env = gym.make(ENV_NAME, render_mode="rgb_array")
    qnet = QNet(n_features=8, n_actions=4)

    dqn_tester = DqnTester(
        env=test_env,
        qnet=qnet,
        env_name=ENV_NAME,
        current_dir=CURRENT_DIR,
    )

    dqn_tester.test()
    test_env.close()


if __name__ == "__main__":
    main()
```

실행 후 다음 위치에 영상이 저장되는지 확인한다.

```text
link_rl/_01_code/_09_DQN/videos/
```

### 5.7 학습 실행

1차 실험:

```python
"learning_rate": 0.001
```

```bash
python dqn_lunar_lander_train.py
```

2차 실험:

```python
"learning_rate": 0.00001
```

```bash
python dqn_lunar_lander_train.py
```

각 실험은 WandB run 이름 또는 tag를 구분한다. 예를 들면 다음처럼 name을 지정하면 비교가 쉽다.

```python
wandb.init(
    project="DQN_LunarLander-v3",
    name=f"lr_{config['learning_rate']}_{self.current_time}",
    config=config,
)
```

단, 기존 코드 구조를 크게 바꾸지 않아도 된다.

### 5.8 테스트 실행

학습이 완료되어 다음 파일이 생성되어 있어야 한다.

```text
models/dqn_LunarLander-v3_latest.pth
```

그 후 다음 명령을 실행한다.

```bash
python dqn_lunar_lander_test.py
```

캡처할 콘솔 예시:

```text
[TOTAL_STEPS: 248, EPISODE REWARD: 239.7]
```

과제에서는 성공 episode 기준으로 reward가 200 이상인 결과를 캡처하는 것이 가장 좋다.

---

## 6. Learning Rate 비교 해석 방향

### 6.1 `[VALIDATION] Mean Episode Reward`

- validation episode에서 greedy policy로 얻은 평균 reward이다.
- 200 이상이면 LunarLander가 해결된 것으로 볼 수 있다.
- `learning_rate=0.001`이 더 빠르게 200에 도달하면 학습 효율이 더 좋다고 해석한다.
- `learning_rate=0.00001`이 너무 느리게 증가하거나 200에 도달하지 못하면 update 크기가 너무 작아 학습이 지연된 것으로 해석한다.

### 6.2 `[TRAIN] Episode Reward`

- 학습 중 각 episode에서 얻은 reward이다.
- 초기에는 무작위 행동으로 인해 음수 reward가 많을 수 있다.
- 학습이 진행되면 추락 감소, 연료 사용 감소, 안정 착륙 증가로 reward가 상승한다.
- 변동성이 큰 이유는 초기 힘, 착륙 위치, 탐험 행동, 환경 dynamics 때문이다.

### 6.3 `[TRAIN] Loss`

- Q-network가 예측한 Q-value와 TD target 사이의 MSE loss이다.
- loss가 낮다고 항상 좋은 정책을 의미하지는 않는다.
- `learning_rate=0.001`에서는 빠른 학습으로 loss 변동이 클 수 있다.
- `learning_rate=0.00001`에서는 loss가 안정적으로 보일 수 있지만 reward 개선이 느리다면 실제 성능은 부족할 수 있다.

### 6.4 `[TRAIN] Epsilon`

- epsilon-greedy 정책에서 random action을 선택할 확률이다.
- 학습 초반에는 exploration을 위해 높고, 학습이 진행되며 감소한다.
- 두 learning rate 실험에서 epsilon schedule이 동일하다면 exploration 조건은 동일하므로, 성능 차이는 주로 learning rate의 영향으로 해석할 수 있다.

### 6.5 `[TRAIN] Replay buffer`

- replay buffer에 저장된 transition 개수이다.
- 학습이 진행되며 증가하고, 최대 용량에 도달하면 오래된 transition이 제거된다.
- buffer가 충분히 쌓이면 다양한 경험을 mini-batch로 샘플링하여 학습 안정성을 높인다.
- 두 learning rate 실험에서 replay buffer 증가 패턴이 유사하다면 데이터 수집량은 비슷하고, 성능 차이는 Q-network 업데이트 속도의 차이로 해석할 수 있다.

### 6.6 최종 선택 문장 예시

실험 결과 `learning_rate=0.001`이 validation mean reward 200 이상에 더 빠르게 도달하고 test episode에서도 200 이상의 reward를 달성했다면 다음처럼 작성한다.

```text
두 learning rate를 비교한 결과, learning_rate=0.001이 더 적합한 것으로 판단된다. learning_rate=0.001은 validation mean episode reward가 더 빠르게 증가하여 해결 기준인 200점에 도달했으며, train episode reward 역시 전반적으로 상승하는 경향을 보였다. 반면 learning_rate=0.00001은 loss 변동이 상대적으로 작을 수 있으나, Q-network의 갱신 폭이 너무 작아 reward 개선 속도가 느리고 동일한 학습 시간 내에 충분한 성능에 도달하지 못했다. 따라서 본 과제에서는 학습 안정성과 수렴 속도를 종합적으로 고려할 때 learning_rate=0.001을 최종 설정으로 선택하였다.
```

반대로 실제 결과에서 `0.00001`이 더 좋다면 위 문장을 반대로 수정한다.

---

## 7. 보고서 구조

보고서는 HWP 또는 MS Word로 작성한 뒤 PDF로 저장한다.

권장 구조:

```text
표지

1. 서론
   1.1 Homework의 목적
   1.2 Q-learning과 DQN의 개요
   1.3 실험 환경 요약

2. 본론
   2.1 Problem 1: Slippery FrozenLake
       2.1.1 환경 설명
       2.1.2 코드 수정 사항
       2.1.3 주요 코드 및 주석
       2.1.4 학습 및 play 결과 캡처
       2.1.5 WandB 그래프와 해석

   2.2 Problem 2: LunarLander
       2.2.1 환경 설명
       2.2.2 DQN 구조 설명
       2.2.3 코드 파일 구성
       2.2.4 주요 코드 및 주석
       2.2.5 learning_rate=0.001 실험 결과
       2.2.6 learning_rate=0.00001 실험 결과
       2.2.7 WandB 그래프 비교 및 해석
       2.2.8 최종 learning rate 선택
       2.2.9 test 실행 결과 캡처

3. 결론
   3.1 과제 수행 결과 요약
   3.2 느낀 점
   3.3 한계 및 개선 방향
```

파일명 예시:

```text
대학원_인공지능특강-1차-이영훈-학번.pdf
```

---

## 8. AI coding tool에게 줄 직접 프롬프트 예시

아래 내용을 Codex 또는 Claude Code에 그대로 붙여 넣고 작업을 시작한다.

```text
You are helping me complete a reinforcement learning homework based on the GitHub repository `linklab/link_rl`.

Repository:
https://github.com/linklab/link_rl

Assignment summary:
1. Problem 1: Run Q-learning on 4x4 FrozenLake-v1 with `is_slippery=True`. Keep the provided config fixed:
   num_episodes=50000, validation_time_steps_interval=1000, validation_num_episodes=10, alpha=0.1, gamma=0.95, epsilon=0.1.
   The existing file is `_01_code/_08_SARSA_Q_Learning/a_frozen_lake_q_learning.py`.
   Verify it uses `IS_SLIPPERY=True`, runs training, logs 4 WandB charts, and performs a final human-render play. Do not change the fixed config.

2. Problem 2: Convert the existing CartPole DQN code in `_01_code/_09_DQN` into LunarLander-v3 DQN code.
   Existing files:
   - c_qnet.py
   - d_dqn_train_test.py
   - e_dqn_cartpole_train.py
   - f_dqn_cartpole_test.py

   Required output files:
   - qnet_lunar_lander.py
   - dqn_train_test.py
   - dqn_lunar_lander_train.py
   - dqn_lunar_lander_test.py

   LunarLander-v3 requirements:
   - env name: `LunarLander-v3`
   - observation dimension: 8
   - action dimension: 4
   - use DQN with replay buffer, target network, epsilon-greedy, validation, model saving/loading, WandB logging
   - fixed config values: `episode_reward_avg_solved=200`, `validation_num_episodes=3`
   - compare two learning rates: 0.001 and 0.00001
   - log these WandB metrics: validation mean episode reward, train episode reward, train loss, train epsilon, replay buffer size

Tasks:
- Inspect the current repository files.
- Create the four required LunarLander files by adapting the existing DQN CartPole implementation.
- Keep the code style close to the course code.
- Do not introduce a completely different RL algorithm.
- Do not use pre-trained models.
- Add clear comments explaining the main code sections.
- Make sure `python dqn_lunar_lander_train.py` trains and saves `models/dqn_LunarLander-v3_latest.pth`.
- Make sure `python dqn_lunar_lander_test.py` loads the saved model and records a test video.
- If there are dependency issues with Box2D, provide the exact installation command.
- After implementation, summarize exactly what files were changed and how to run the experiments.
```

---

## 9. 디버깅 체크리스트

### LunarLander import 오류

증상:

```text
ModuleNotFoundError: No module named 'Box2D'
```

해결:

```bash
pip install "gymnasium[box2d]"
```

### WandB 로그인 오류

해결:

```bash
wandb login
```

또는 임시로:

```python
use_wandb = False
```

단, 과제에서 WandB 그래프가 필요하므로 최종 실험은 WandB를 켜고 실행한다.

### 모델 파일 없음

증상:

```text
FileNotFoundError: dqn_LunarLander-v3_latest.pth
```

원인:

- 학습이 아직 solved 기준에 도달하지 못해 model save가 실행되지 않음.
- `ENV_NAME`이 train과 test에서 다름.

해결:

- train을 더 오래 실행한다.
- train과 test 모두 `ENV_NAME = "LunarLander-v3"`인지 확인한다.
- `models/` 폴더에 실제 저장된 파일명을 확인한다.

### Test reward가 200 미만

가능한 원인:

- 학습이 충분하지 않음.
- 저장된 latest 모델이 낮은 reward의 모델임.
- epsilon을 test에서 0으로 두지 않음.

확인:

```python
action = self.qnet.get_action(observation, epsilon=0.0)
```

### FrozenLake play 실패

원인:

- stochastic 환경 특성상 학습된 policy도 한 번의 play에서 실패할 수 있음.

해결:

- 다시 실행하여 `PLAY EPISODE SUCCESS!!!` 화면을 확보한다.
- config는 바꾸지 않는다.

---

## 10. 최종 제출 전 체크리스트

- [ ] Problem 1 `IS_SLIPPERY=True` 확인
- [ ] Problem 1 고정 config 미수정 확인
- [ ] Problem 1 학습 완료 콘솔 캡처
- [ ] Problem 1 play 성공 화면 캡처
- [ ] Problem 1 WandB 그래프 4개 캡처
- [ ] Problem 1 그래프 4개 해석 작성
- [ ] Problem 2 required file 4개 존재 확인
- [ ] Problem 2 `LunarLander-v3` 사용 확인
- [ ] Problem 2 `QNet(n_features=8, n_actions=4)` 확인
- [ ] Problem 2 `validation_num_episodes=3` 확인
- [ ] Problem 2 `episode_reward_avg_solved=200` 확인
- [ ] Problem 2 learning rate 0.001 실험 완료
- [ ] Problem 2 learning rate 0.00001 실험 완료
- [ ] Problem 2 WandB 그래프 5개 캡처
- [ ] Problem 2 그래프 5개 해석 작성
- [ ] 더 좋은 learning rate 선택 및 이유 작성
- [ ] `dqn_lunar_lander_test.py` 성공 실행 콘솔 캡처
- [ ] LunarLander test 화면 또는 영상 캡처
- [ ] 보고서가 표지, 서론, 본론, 결론 구조를 가짐
- [ ] PDF로 저장
- [ ] 파일명 규칙 확인
- [ ] 제출 마감: 2026-05-05 23:59:59

---

## 11. 보고서 결론 작성 예시

```text
본 과제에서는 discrete state space를 갖는 FrozenLake 환경에 Q-learning을 적용하고, continuous state space를 갖는 LunarLander 환경에 DQN을 적용하였다. FrozenLake에서는 slippery 설정으로 인해 동일한 action을 선택하더라도 실제 이동 방향이 확률적으로 달라지는 stochastic 특성이 나타났으며, Q-learning은 반복적인 경험을 통해 각 state-action pair의 기대 가치를 학습하였다. LunarLander에서는 8차원 연속 관측값을 직접 Q-table로 표현하기 어렵기 때문에 neural network를 이용하여 Q-value를 근사하는 DQN을 사용하였다. 또한 replay buffer와 target network를 통해 학습 안정성을 확보하였다.

Learning rate 비교 결과, 실험에서 더 높은 validation mean episode reward와 안정적인 test reward를 보인 learning rate를 최종 설정으로 선택하였다. 이를 통해 learning rate가 너무 작으면 학습 속도가 느려지고, 너무 크면 loss와 reward 변동성이 커질 수 있음을 확인하였다. 강화학습 실험은 환경의 stochasticity와 exploration 정책의 영향을 크게 받기 때문에 단일 episode 결과보다 validation 평균 reward와 반복 실험 결과를 함께 해석하는 것이 중요하다는 점을 확인하였다.
```
