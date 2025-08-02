import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import FarmAIEnv

def run_ppo_model(render=True):
    # Load the trained model
    model_path = "models/ppo/ppo_model"
    model = PPO.load(model_path)

    # Create environment (same as training)
    env = FarmAIEnv(render_mode="human" if render else None)
    env = DummyVecEnv([lambda: env])

    # Run the agent
    obs = env.reset()
    for _ in range(1000):  # Run for 1000 steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    run_ppo_model(render=True)  # Set render=False for faster execution