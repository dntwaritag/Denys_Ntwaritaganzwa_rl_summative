import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from training import dqn_training, ppo_training, reinforce_training, actor_critic_training
from environment.rendering import generate_random_actions_gif, record_trained_agent
from evaluation.evaluate_models import evaluate_model, compare_models

def setup_directories():
    """Create all required directories with robust error handling"""
    required_dirs = [
        "models/dqn",
        "models/ppo", 
        "models/a2c",
        "models/reinforce",
        "results/plots",
        "results/gifs",
        "results/videos",
        "results/metrics",
        "results/comparisons"
    ]
    
    for dir_path in required_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except FileExistsError:
            if os.path.isfile(dir_path):
                os.remove(dir_path)
                os.makedirs(dir_path)

def save_training_metrics(algorithm, rewards, timesteps, hyperparams):
    """Save complete training metrics and plots with error handling"""
    # Ensure metrics directory exists
    os.makedirs(f"results/metrics", exist_ok=True)
    os.makedirs(f"results/plots", exist_ok=True)
    
    # Validate inputs
    if len(rewards) == 0 or len(timesteps) == 0:
        print(f"Warning: No training metrics available for {algorithm}")
        return
    
    # Save numerical metrics
    metrics = {
        'rewards': np.array(rewards),
        'timesteps': np.array(timesteps),
        'hyperparameters': hyperparams
    }
    np.savez(f"results/metrics/{algorithm}_metrics.npz", **metrics)
    
    try:
        # Create training progress plot
        plt.figure(figsize=(12, 6))
        
        # Reward progression
        plt.subplot(1, 2, 1)
        plt.plot(timesteps, rewards)
        plt.title(f"{algorithm} Training Performance")
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward")
        plt.grid(True)
        
        # Moving average (only if we have enough data points)
        if len(rewards) > 10:
            window = min(50, len(rewards)//4)  # Dynamic window size
            weights = np.ones(window)/window
            moving_avg = np.convolve(rewards, weights, 'valid')
            
            plt.subplot(1, 2, 2)
            plt.plot(timesteps[window-1:], moving_avg)
            plt.title(f"{algorithm} Moving Average (window={window})")
            plt.xlabel("Timesteps")
            plt.ylabel("Avg Reward")
            plt.grid(True)
        else:
            # If not enough data, just show empty right plot
            plt.subplot(1, 2, 2)
            plt.text(0.5, 0.5, 'Not enough data\nfor moving average', 
                    ha='center', va='center')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"results/plots/{algorithm}_training.png", dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error saving {algorithm} training metrics: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="Farm AI RL Agent - Value vs Policy Methods Comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training configuration
    parser.add_argument('--train', choices=['dqn', 'ppo', 'reinforce', 'a2c', 'all'],
                      help="Train RL models")
    parser.add_argument('--episodes', type=int, default=1000,
                      help="Training episodes")
    
    # Evaluation and visualization
    parser.add_argument('--evaluate', choices=['dqn', 'ppo', 'reinforce', 'a2c'],
                      help="Evaluate specific model")
    parser.add_argument('--compare', action='store_true',
                      help="Compare all trained models")
    parser.add_argument('--visualize', action='store_true',
                      help="Generate GIF of random actions")
    parser.add_argument('--record', choices=['dqn', 'ppo', 'reinforce', 'a2c'],
                      help="Record video of trained agent")
    parser.add_argument('--render', action='store_true',
                      help="Render environment during evaluation")
    
    args = parser.parse_args()
    setup_directories()

    # Task 1: Environment Visualization
    if args.visualize:
        print("\n=== Generating Environment Visualization ===")
        gif_path = "results/gifs/environment_random_actions.gif"
        generate_random_actions_gif(filename=gif_path, steps=300, fps=30)
        print(f"Saved visualization to {gif_path}")

    # Task 2: Train RL Models
    if args.train:
        train_args = {
            'n_episodes': args.episodes,
            'render': args.render
        }
        
        if args.train in ['dqn', 'all']:
            print("\n=== Training DQN (Value-Based) ===")
            rewards, timesteps, params = dqn_training.train_dqn(**train_args)
            save_training_metrics("DQN", rewards, timesteps, params)
        
        if args.train in ['ppo', 'all']:
            print("\n=== Training PPO (Policy Gradient) ===")
            rewards, timesteps, params = ppo_training.train_ppo(**train_args)
            save_training_metrics("PPO", rewards, timesteps, params)
        
        if args.train in ['reinforce', 'all']:
            print("\n=== Training REINFORCE (Policy Gradient) ===")
            rewards, timesteps, params = reinforce_training.train_reinforce(**train_args)
            save_training_metrics("REINFORCE", rewards, timesteps, params)
        
        if args.train in ['a2c', 'all']:
            print("\n=== Training Actor-Critic (Policy Gradient) ===")
            rewards, timesteps, params = actor_critic_training.train_actor_critic(**train_args)
            save_training_metrics("A2C", rewards, timesteps, params)

    # Task 3: Evaluate Models
    if args.evaluate:
        print(f"\n=== Evaluating {args.evaluate.upper()} Model ===")
        metrics = evaluate_model(algorithm=args.evaluate, n_episodes=3, render=args.render)
        
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric:>20}: {value:.2f}")

    # Task 4: Model Comparison
    if args.compare:
        print("\n=== Comparing All Trained Models ===")
        results = compare_models(n_episodes=10)
        
        print("\nPerformance Comparison:")
        print("{:<12} {:<12} {:<12} {:<12}".format(
            "Algorithm", "Avg Reward", "Success Rate", "Steps/Episode"))
        
        for algo, res in results.items():
            print("{:<12} {:<12.2f} {:<12.2%} {:<12.1f}".format(
                algo, res['avg_reward'], res['success_rate'], res['avg_steps']))

    # Task 5: Record Agent Video
    if args.record:
        print(f"\n=== Recording {args.record.upper()} Agent ===")
        video_path = f"{args.record}_agent"
        record_trained_agent(
            model_type=args.record,  # Changed to match parameter name
            output_path=video_path,
            episode_length=180,  # 3 minute video
            fps=30
        )
        print(f"Saved agent video to results/videos/{video_path}.mp4")

if __name__ == "__main__":
    main()