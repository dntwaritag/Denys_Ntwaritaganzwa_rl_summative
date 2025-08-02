import os
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from environment.custom_env import FarmAIEnv
import numpy as np
import matplotlib.pyplot as plt

class TrainingCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self):
        # Collect rewards from info dict if available
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
        
        # Save model periodically
        if self.n_calls % self.save_freq == 0:
            self.model.save(f"{self.save_path}/a2c_{self.n_calls}")
        return True

def train_actor_critic(n_episodes=1000, render=False):
    """Train an A2C agent with parameters matching main.py interface"""
    # Setup directories
    os.makedirs("models/a2c", exist_ok=True)
    os.makedirs("results/a2c", exist_ok=True)

    # Create environment
    env = FarmAIEnv(render_mode="human" if render else None)

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=0.0007,
        gamma=0.99,
        verbose=1
    )
    
    callback = TrainingCallback(
        save_freq=10000, 
        save_path="models/a2c"
    )
    
    model.learn(total_timesteps=n_episodes*1000, callback=callback)
    model.save("models/a2c/a2c_model")
    
    # Get training metrics
    rewards = np.array(callback.episode_rewards)
    timesteps = np.arange(len(rewards)) * 1000  # Approximate timesteps

    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards)
    plt.title("A2C Training Performance")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig("results/a2c/a2c_training.png")
    plt.close()

    return rewards, timesteps, model.get_parameters()

if __name__ == "__main__":
    train_actor_critic()