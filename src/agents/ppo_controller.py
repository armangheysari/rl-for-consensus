
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ..env.consensus_env import ConsensusEnv


class RewardLoggerCallback(BaseCallback):

    def __init__(self, log_interval: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_count = 0
        self.recent_rewards: list[float] = []

    def _on_step(self) -> bool:
        # SB3 stores episode info in self.locals["infos"]
        for info in self.locals.get("infos", []):
            if "stats" not in info:
                continue
            # Use the actual reward from the environment (which now includes
            # the efficiency penalty) instead of recomputing it. SB3 stores
            # the per-step reward in self.locals["rewards"].
            pass
        # SB3 also accumulates rewards per step; we read the actual reward
        # from the environment step. Since our env returns one reward per
        # window (one step), we can just look at the latest reward.
        rewards = self.locals.get("rewards", [])
        if len(rewards) > 0:
            self.recent_rewards.append(float(rewards[-1]))
            self.episode_count += 1
            if self.episode_count % self.log_interval == 0:
                window = self.recent_rewards[-self.log_interval:]
                mean_r = sum(window) / len(window)
                if self.verbose > 0:
                    print(f"  step {self.num_timesteps}: episode {self.episode_count}: mean window reward = {mean_r:.3f}")
        return True


class PPOController:

    def __init__(
        self,
        env: ConsensusEnv,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        gamma: float = 0.99,
        ent_coef: float = 0.01,
        seed: int = 0,
    ):
        # SB3 expects a vectorized env; DummyVecEnv wraps a single env
        # without any actual parallelism, which is fine for our small env.
        self.env = env
        self.vec_env = DummyVecEnv([lambda: env])
        self.model = PPO(
            policy="MlpPolicy",
            env=self.vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            verbose=0,
            seed=seed,
        )
        self.seed = seed

    def train(self, total_timesteps: int = 50_000, log_interval: int = 20) -> None:
        callback = RewardLoggerCallback(log_interval=log_interval, verbose=1)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    @classmethod
    def load(cls, path: str | Path, env: ConsensusEnv) -> "PPOController":
        controller = cls.__new__(cls)
        controller.env = env
        controller.vec_env = DummyVecEnv([lambda: env])
        controller.model = PPO.load(str(path), env=controller.vec_env)
        controller.seed = 0
        return controller

    def act(self, obs) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
