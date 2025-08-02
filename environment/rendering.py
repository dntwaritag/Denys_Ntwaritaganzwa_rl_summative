import pygame
import imageio
import numpy as np
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from environment.custom_env import FarmAIEnv

def generate_random_actions_gif(filename="results/gifs/random_actions.gif", steps=300, fps=30):
    """Generate GIF of random agent actions"""
    try:
        env = FarmAIEnv(render_mode="rgb_array")
        frames = []
        obs, _ = env.reset()
        
        for _ in range(steps):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            if terminated or truncated:
                obs, _ = env.reset()
        
        if frames:
            imageio.mimsave(filename, frames, fps=fps)
            print(f"Saved GIF to {filename}")
        env.close()
    except Exception as e:
        print(f"Error generating GIF: {str(e)}")
        if 'env' in locals():
            env.close()

def record_trained_agent(model_type, output_path="results/videos/agent_recording", 
                        episode_length=180, fps=30):
    """Record video of trained agent"""
    try:
        # Initialize environment
        env = DummyVecEnv([lambda: FarmAIEnv(render_mode="rgb_array")])
        
        # Load appropriate model
        if model_type == "dqn":
            from stable_baselines3 import DQN
            model = DQN.load("models/dqn/dqn_model")
        elif model_type == "ppo":
            from stable_baselines3 import PPO
            model = PPO.load("models/ppo/ppo_model")
        elif model_type == "a2c":
            from stable_baselines3 import A2C
            model = A2C.load("models/a2c/a2c_model")
        elif model_type == "reinforce":
            raise NotImplementedError("REINFORCE video recording not implemented")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Setup video recording
        env = VecVideoRecorder(
            env,
            "results/videos",
            record_video_trigger=lambda x: x == 0,
            video_length=episode_length,
            name_prefix=output_path
        )

        # Run and record
        obs = env.reset()
        for _ in range(episode_length):
            action, _ = model.predict(obs)
            obs, _, _, _ = env.step(action)

        env.close()
        print(f"Successfully saved video to {output_path}.mp4")
    except Exception as e:
        print(f"Error recording video: {str(e)}")
        if 'env' in locals():
            env.close()