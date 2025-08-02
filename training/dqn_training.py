import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import FarmAIEnv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_dqn(n_episodes=1000, render=False):
    """Train a DQN agent with parameters matching main.py interface"""
    # Setup directories
    os.makedirs("logs/dqn", exist_ok=True)
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("results/dqn", exist_ok=True)

    # Create environment
    env = FarmAIEnv(render_mode="human" if render else None)
    env = Monitor(env, "logs/dqn")
    env = DummyVecEnv([lambda: env])

    # DQN model configuration
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="logs/dqn"
    )

    # Train the model
    model.learn(total_timesteps=n_episodes*1000)  # Assuming ~1000 steps per episode
    
    # Save model
    model.save("models/dqn/dqn_model")

    # Get training metrics
    df = pd.read_csv("logs/dqn/monitor.csv", skiprows=1)
    rewards = df['r'].values
    timesteps = df['l'].cumsum().values

    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards)
    plt.title("DQN Training Performance")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig("results/dqn/dqn_training.png")
    plt.close()

    return rewards, timesteps, model.get_parameters()

if __name__ == "__main__":
    train_dqn()