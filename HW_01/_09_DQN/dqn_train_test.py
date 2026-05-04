# DQN Trainer and Tester — adapted from d_dqn_train_test.py for LunarLander
# Key differences from the CartPole original:
#   - WandB run name includes learning_rate for easy comparison between lr=0.001 and lr=0.00001 runs
#   - Fixed typo in DqnTester.test(): "TOAL_STEPS" -> "TOTAL_STEPS"
import os
import time
from datetime import datetime
from shutil import copyfile

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
import collections

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = collections.namedtuple(
    typename="Transition", field_names=["observation", "action", "next_observation", "reward", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def size(self) -> int:
        return len(self.buffer)

    def append(self, transition: Transition):
        self.buffer.append(transition)

    def pop(self) -> Transition:
        return self.buffer.pop()

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        observations, actions, next_observations, rewards, dones = zip(*[self.buffer[idx] for idx in indices])

        observations = np.array(observations)
        next_observations = np.array(next_observations)

        actions = np.array(actions)
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions
        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards
        dones = np.array(dones, dtype=bool)

        observations = torch.tensor(observations, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

        return observations, actions, next_observations, rewards, dones


class DqnTrainer:
    def __init__(
            self, env: gym.Env, valid_env: gym.Env, qnet, target_qnet, config: dict, use_wandb: bool, current_dir
    ):
        self.env = env
        self.valid_env = valid_env
        self.use_wandb = use_wandb
        self.config = config

        self.model_dir = os.path.join(current_dir, "models")
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        if self.use_wandb:
            # Include learning_rate in run name so the two experiments are distinguishable in WandB
            run_name = "lr_{0}_{1}".format(self.config["learning_rate"], self.current_time)
            self.wandb = wandb.init(
                project="DQN_{0}".format(self.config["env_name"]),
                name=run_name,
                config=config
            )

        self.epsilon_scheduled_last_episode = self.config["max_num_episodes"] * self.config["epsilon_final_scheduled_percent"]

        self.qnet = qnet
        self.target_qnet = target_qnet
        self.target_qnet.load_state_dict(self.qnet.state_dict())

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.config["learning_rate"])

        self.replay_buffer = ReplayBuffer(self.config["replay_buffer_size"])

        self.time_steps = 0
        self.training_time_steps = 0

        self.total_train_start_time = None

    def epsilon_scheduled(self, current_episode: int) -> float:
        fraction = min(current_episode / self.epsilon_scheduled_last_episode, 1.0)
        epsilon_span = self.config["epsilon_start"] - self.config["epsilon_end"]
        epsilon = min(self.config["epsilon_start"] - fraction * epsilon_span, self.config["epsilon_start"])
        return epsilon

    def train_loop(self):
        loss = 0.0

        self.total_train_start_time = time.time()

        validation_episode_reward_avg = None

        is_terminated = False

        for n_episode in range(1, self.config["max_num_episodes"] + 1):
            epsilon = self.epsilon_scheduled(n_episode)

            episode_reward = 0

            observation, _ = self.env.reset()

            done = False

            while not done:
                self.time_steps += 1

                action = self.qnet.get_action(observation, epsilon)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.replay_buffer.append(transition)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

                if self.time_steps % self.config["steps_between_train"] == 0 and self.time_steps > self.config["batch_size"]:
                    loss = self.train()

                if self.time_steps % self.config["validation_time_steps_interval"] == 0:
                    validation_episode_reward_lst, validation_episode_reward_avg = self.validate()

                    if validation_episode_reward_avg > self.config["episode_reward_avg_solved"]:
                        print("Solved in {0:,} time steps ({1:,} training steps)!".format(
                            self.time_steps, self.training_time_steps
                        ))
                        self.model_save(validation_episode_reward_avg)
                        is_terminated = True
                        break

            if validation_episode_reward_avg is None:
                validation_episode_reward_avg = episode_reward

            if n_episode % self.config["print_episode_interval"] == 0:
                print(
                    "[Episode {:3,}, Time Steps {:6,}]".format(n_episode, self.time_steps),
                    "Episode Reward: {:>7.3f},".format(episode_reward),
                    "Replay buffer: {:>6,},".format(self.replay_buffer.size()),
                    "Loss: {:6.4f},".format(loss),
                    "Epsilon: {:4.2f},".format(epsilon),
                    "Training Steps: {:5,},".format(self.training_time_steps),
                )

            if self.use_wandb:
                self.wandb.log(
                    {
                        "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(
                            self.config["validation_num_episodes"]
                        ): validation_episode_reward_avg,
                        "[TRAIN] Episode Reward": episode_reward,
                        "[TRAIN] Loss": loss if loss != 0.0 else 0.0,
                        "[TRAIN] Epsilon": epsilon,
                        "[TRAIN] Replay buffer": self.replay_buffer.size(),
                        "Training Episode": n_episode,
                        "Training Steps": self.training_time_steps,
                    }
                )

            if is_terminated:
                break

        total_training_time = time.time() - self.total_train_start_time
        total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))
        if self.use_wandb:
            self.wandb.finish()

    def train(self) -> float:
        self.training_time_steps += 1

        batch = self.replay_buffer.sample(self.config["batch_size"])
        observations, actions, next_observations, rewards, dones = batch

        q_out = self.qnet(observations)
        q_values = q_out.gather(dim=-1, index=actions)

        with torch.no_grad():
            q_prime_out = self.target_qnet(next_observations)
            max_q_prime = q_prime_out.max(dim=1, keepdim=True).values
            max_q_prime[dones] = 0.0
            targets = rewards + self.config["gamma"] * max_q_prime

        loss = F.mse_loss(targets.detach(), q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.time_steps % self.config["target_sync_time_steps_interval"] == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

        return loss.item()

    def model_save(self, validation_episode_reward_avg: float):
        filename = "dqn_{0}_{1:4.1f}_{2}.pth".format(
            self.config["env_name"], validation_episode_reward_avg, self.current_time
        )
        torch.save(self.qnet.state_dict(), os.path.join(self.model_dir, filename))

        # lr-specific latest file: prevents the two learning-rate experiments from overwriting each other
        # e.g. dqn_LunarLander-v3_lr_0.001_latest.pth  /  dqn_LunarLander-v3_lr_0.00001_latest.pth
        lr_latest = "dqn_{0}_lr_{1}_latest.pth".format(
            self.config["env_name"], self.config["learning_rate"]
        )
        copyfile(
            src=os.path.join(self.model_dir, filename),
            dst=os.path.join(self.model_dir, lr_latest)
        )

    def validate(self) -> tuple[np.ndarray, float]:
        episode_reward_lst = np.zeros(shape=(self.config["validation_num_episodes"],), dtype=float)

        for i in range(self.config["validation_num_episodes"]):
            episode_reward = 0

            observation, _ = self.valid_env.reset()

            done = False

            while not done:
                action = self.qnet.get_action(observation, epsilon=0.0)

                next_observation, reward, terminated, truncated, _ = self.valid_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward

        episode_reward_avg = np.average(episode_reward_lst)

        total_training_time = time.time() - self.total_train_start_time
        total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))

        print(
            "[Validation Episode Reward: {0}] Average: {1:.3f}, Elapsed Time: {2}".format(
                episode_reward_lst, episode_reward_avg, total_training_time
            )
        )
        return episode_reward_lst, episode_reward_avg


class DqnTester:
    def __init__(self, env: gym.Env, qnet, env_name, current_dir, model_path: str = None):
        self.env = env

        self.model_dir = os.path.join(current_dir, "models")
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.video_dir = os.path.join(current_dir, "videos")
        if not os.path.exists(self.video_dir):
            os.mkdir(self.video_dir)

        self.env = gym.wrappers.RecordVideo(
            env=self.env, video_folder=self.video_dir,
            name_prefix="dqn_{0}_test_video".format(env_name)
        )

        self.qnet = qnet

        # model_path overrides the default filename; set it in dqn_lunar_lander_test.py
        if model_path is None:
            model_path = os.path.join(self.model_dir, "dqn_{0}_latest.pth".format(env_name))
        print("Loading model from:", model_path)
        model_params = torch.load(model_path, weights_only=True, map_location=DEVICE)
        self.qnet.load_state_dict(model_params)
        self.qnet.eval()

    def test(self):
        episode_reward = 0

        observation, _ = self.env.reset()
        time_steps = 0

        done = False

        while not done:
            time_steps += 1
            # epsilon=0.0: pure greedy policy for evaluation (no random exploration)
            action = self.qnet.get_action(observation, epsilon=0.0)

            next_observation, reward, terminated, truncated, _ = self.env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        self.env.close()
        # Fixed typo from original: "TOAL_STEPS" -> "TOTAL_STEPS"
        print("[TOTAL_STEPS: {0:3d}, EPISODE REWARD: {1:4.1f}]".format(time_steps, episode_reward))
