import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import FarmAIEnv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_ppo(n_episodes=1000, render=False):
    """Train a PPO agent with parameters matching main.py interface"""
    # Setup directories
    os.makedirs("logs/ppo", exist_ok=True)
    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("results/ppo", exist_ok=True)

    # Create environment
    env = FarmAIEnv(render_mode="human" if render else None)
    env = Monitor(env, "logs/ppo")
    env = DummyVecEnv([lambda: env])

    # PPO model configuration
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="logs/ppo"
    )

    # Train the model
    model.learn(total_timesteps=n_episodes*1000)
    
    # Save model
    model.save("models/ppo/ppo_model")

    # Get training metrics
    df = pd.read_csv("logs/ppo/monitor.csv", skiprows=1)
    rewards = df['r'].values
    timesteps = df['l'].cumsum().values

    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards)
    plt.title("PPO Training Performance")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig("results/ppo/ppo_training.png")
    plt.close()

    return rewards, timesteps, model.get_parameters()

if __name__ == "__main__":
    train_ppo()