import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import FarmAIEnv
import torch
from training.reinforce_training import PolicyNetwork

def evaluate_model(model_path, algorithm, n_episodes=10, render=False):
    """Evaluate a specific model"""
    env = FarmAIEnv(render_mode="human" if render else None)
    obs_shape = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Load model based on algorithm type
    if algorithm == 'dqn':
        model = DQN.load(model_path, env=env)
    elif algorithm == 'ppo':
        model = PPO.load(model_path, env=env)
    elif algorithm == 'a2c':
        model = A2C.load(model_path, env=env)
    elif algorithm == 'reinforce':
        policy_net = PolicyNetwork(obs_shape, action_dim)
        policy_net.load_state_dict(torch.load(model_path))
        model = policy_net
    
    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = False
        total_reward = 0
        steps = 0
        
        while not terminated:
            if algorithm == 'reinforce':
                action_probs = model(torch.FloatTensor(obs))
                action = torch.argmax(action_probs).item()
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, _, _ = env.step(action)
            total_reward += reward
            steps += 1
            if render:
                env.render()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    env.close()
    
    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_steps': np.mean(episode_lengths),
        'success_rate': np.mean(np.array(episode_rewards) > 0)  # Assuming positive reward means success
    }

def compare_models(n_episodes=10, render=False):
    """Compare performance of all trained models"""
    models = {
        "DQN": ("models/dqn/dqn_model.zip", 'dqn'),
        "PPO": ("models/ppo/ppo_model.zip", 'ppo'),
        "A2C": ("models/a2c/a2c_model.zip", 'a2c'),
        "REINFORCE": ("models/reinforce/reinforce_model.pth", 'reinforce')
    }
    
    results = {}
    
    for name, (path, algo) in models.items():
        if not os.path.exists(path):
            print(f"Warning: Model not found at {path}")
            continue
            
        print(f"\nEvaluating {name} model...")
        try:
            metrics = evaluate_model(path, algo, n_episodes=n_episodes, render=render)
            results[name] = metrics
            print(f"{name} Results:")
            print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
            print(f"  Success Rate: {metrics['success_rate']:.2%}")
            print(f"  Avg Steps/Episode: {metrics['avg_steps']:.1f}")
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            continue
    
    # Generate comparison plot
    if results:
        plt.figure(figsize=(12, 6))
        
        # Reward comparison
        plt.subplot(1, 2, 1)
        names = list(results.keys())
        rewards = [res['avg_reward'] for res in results.values()]
        plt.bar(names, rewards)
        plt.title("Average Reward Comparison")
        plt.ylabel("Reward")
        
        # Success rate comparison
        plt.subplot(1, 2, 2)
        success_rates = [res['success_rate'] for res in results.values()]
        plt.bar(names, success_rates)
        plt.title("Success Rate Comparison")
        plt.ylabel("Success Rate")
        
        plt.tight_layout()
        os.makedirs("results/comparisons", exist_ok=True)
        plt.savefig("results/comparisons/model_comparison.png", dpi=300)
        plt.close()
    
    return results

if __name__ == "__main__":
    compare_models()