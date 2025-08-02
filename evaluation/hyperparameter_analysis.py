import numpy as np
import matplotlib.pyplot as plt
from training.dqn_training import train_dqn
from environment.custom_env import RescueEnv

def analyze_hyperparameters():
    # Test different learning rates
    lrs = [1e-4, 1e-3, 1e-2]
    results = {}
    
    for lr in lrs:
        print(f"\nTraining DQN with learning_rate={lr}")
        env = RescueEnv()
        env = DummyVecEnv([lambda: env])
        
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=lr,
            # Other parameters kept constant...
            verbose=0
        )
        
        model.learn(total_timesteps=20000)
        
        # Evaluate
        rewards = []
        for _ in range(20):
            obs = env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
            rewards.append(total_reward)
        
        results[lr] = np.mean(rewards)
        env.close()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(list(results.keys()), list(results.values()), 'o-')
    plt.xscale('log')
    plt.xlabel("Learning Rate")
    plt.ylabel("Average Reward")
    plt.title("DQN Performance vs. Learning Rate")
    plt.grid()
    plt.savefig("results/hyperparameter_analysis.png")
    plt.close()
    
    return results

if __name__ == "__main__":
    analyze_hyperparameters()