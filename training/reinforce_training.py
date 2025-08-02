import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
from environment.custom_env import FarmAIEnv
import os
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)

def train_reinforce(n_episodes=1000, render=False):
    """Train a REINFORCE agent with parameters matching main.py interface"""
    # Setup directories
    os.makedirs("models/reinforce", exist_ok=True)
    os.makedirs("results/reinforce", exist_ok=True)

    env = FarmAIEnv(render_mode="human" if render else None)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = PolicyNetwork(input_dim, output_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.0003)
    gamma = 0.99
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        
        terminated = False
        while not terminated:
            state_tensor = torch.FloatTensor(state)
            action_probs = agent(state_tensor)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            next_state, reward, terminated, _, _ = env.step(action.item())
            
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        # Normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Update policy
        optimizer.zero_grad()
        policy_loss = [-log_prob * G for log_prob, G in zip(log_probs, returns)]
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()
        
        # Record episode reward
        episode_rewards.append(sum(rewards))
    
    # Save model
    torch.save(agent.state_dict(), "models/reinforce/reinforce_model.pth")
    
    # Get training metrics
    rewards = np.array(episode_rewards)
    timesteps = np.arange(n_episodes) * 1000  # Approximate timesteps

    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards)
    plt.title("REINFORCE Training Performance")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig("results/reinforce/reinforce_training.png")
    plt.close()

    return rewards, timesteps, agent.state_dict()

if __name__ == "__main__":
    train_reinforce()