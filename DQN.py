# ================================================================
# Vanilla DQN (Deep Q-Network)
# ================================================================
# This script trains a standard DQN agent on the CartPole-v1 environment.
#
# Key Features:
#  - Fully connected Q-network (no additional encoders or bonuses)
#  - Experience replay buffer for stable training
#  - Epsilon‑greedy exploration with decay
#  - Target network with periodic hard updates
#  - Gradient clipping for stability
#  - Plots episode rewards and training losses over time
#
# Authors: Yash Kini, Shiv Davay, Shreya Polavarapu
# ================================================================
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# ===============================================================
# Reproducibility
# ===============================================================
# Fix random seeds so runs are as deterministic as possible.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===============================================================
# Q-Network Definition
# ===============================================================
class DQN(nn.Module):
    """
    A simple fully-connected Deep Q-Network (DQN):
    - Input: raw state vector (e.g., [cart position, velocity, angle, angular velocity])
    - Output: Q-values for each possible action in the environment.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        # Forward pass: return Q-values for all actions
        return self.net(x)

# ===============================================================
# Utility Functions
# ===============================================================
def epsilon_greedy(model, state, epsilon, action_space):
    """
    Select an action using epsilon-greedy exploration:
    - With probability epsilon, pick a random action.
    - Otherwise, pick the action with the highest predicted Q-value.
    """
    if random.random() < epsilon:
        return random.randrange(action_space)
    with torch.no_grad():
        q_values = model(state)
        return int(torch.argmax(q_values).item())

def soft_update(target, source, tau=0.005):
    """
    Perform a soft update of the target network parameters:
    θ_target = τ * θ_source + (1 - τ) * θ_target
    This slowly tracks the learned network to stabilize training.
    """
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

# ===============================================================
# Training Loop
# ===============================================================
def train_dqn(env, model, target_model, optimizer,
              num_episodes=800, gamma=0.99,
              epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995,
              batch_size=128, buffer_size=100000, warmup_steps=1000,
              target_update_freq=1000):
    """
    Train a vanilla DQN agent on the given environment.
    Args:
        env: Gym environment (e.g., CartPole-v1)
        model: Q-network (online network)
        target_model: Target Q-network
        optimizer: Optimizer for training Q-network
        num_episodes: Number of training episodes
        gamma: Discount factor
        epsilon_start: Initial exploration probability
        epsilon_min: Minimum exploration probability
        epsilon_decay: Decay factor per episode for epsilon
        batch_size: Size of minibatch for training
        buffer_size: Max size of experience replay buffer
        warmup_steps: Steps before learning starts
        target_update_freq: How often to update target network
    Returns:
        rewards_history: List of total rewards per episode
        losses_history: List of training losses over time
    """
    replay_buffer = deque(maxlen=buffer_size)
    rewards_history, losses_history = [], []
    epsilon = epsilon_start
    step_count = 0

    # Main training loop
    for episode in range(num_episodes):
        state, _ = env.reset(seed=SEED)
        state = torch.FloatTensor(state).unsqueeze(0)
        done, total_reward = False, 0.0

        while not done:
            step_count += 1

            # Select an action using epsilon-greedy
            action = epsilon_greedy(model, state, epsilon, env.action_space.n)

            # Step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0)

            # Store transition in replay buffer
            replay_buffer.append((state, action, reward, next_state_t, done))
            state = next_state_t

            # Only start learning after enough experiences are collected
            if len(replay_buffer) >= warmup_steps:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards_b, next_states, dones = zip(*batch)

                # Combine into tensors
                states = torch.cat(states)
                next_states = torch.cat(next_states)
                actions = torch.LongTensor(actions)
                rewards_b = torch.FloatTensor(rewards_b)
                dones = torch.FloatTensor(dones)

                # Compute target Q-values using target network
                with torch.no_grad():
                    q_next = target_model(next_states).max(1)[0]
                    q_target = rewards_b + gamma * q_next * (1 - dones)

                # Get Q-values for taken actions
                q_values = model(states).gather(1, actions.view(-1, 1)).squeeze(1)

                # Compute loss (Mean Squared Error)
                loss = (q_values - q_target).pow(2).mean()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
                optimizer.step()

                # Record loss
                losses_history.append(loss.item())

                # Update target network periodically (hard update)
                if step_count % target_update_freq == 0:
                    target_model.load_state_dict(model.state_dict())

        # Episode completed
        rewards_history.append(total_reward)
        # Decay epsilon after each episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            mean_reward = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
            mean_loss = np.mean(losses_history[-100:]) if losses_history else 0.0
            print(f"[Ep {episode+1:03d}] Reward: {total_reward:.2f} | "
                  f"Mean(50): {mean_reward:.2f} | Eps: {epsilon:.3f} | Loss: {mean_loss:.4f}")

    return rewards_history, losses_history

# ===============================================================
# Main Entry Point
# ===============================================================
def main():
    # Create CartPole environment
    env = gym.make("CartPole-v1")

    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create main and target Q-networks
    model = DQN(state_dim, action_dim)
    target_model = DQN(state_dim, action_dim)
    target_model.load_state_dict(model.state_dict())  # Sync initially

    # Optimizer for Q-network
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the agent
    rewards, losses = train_dqn(env, model, target_model, optimizer)

    # Smoothing function for nicer plots
    def smooth(data, w=0.9):
        if not data:
            return []
        smoothed, last = [], data[0]
        for x in data:
            last = last * w + (1 - w) * x
            smoothed.append(last)
        return smoothed

    # Plot training results
    plt.figure(figsize=(12, 5))
    # Rewards over episodes
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Raw Reward", color='royalblue', alpha=0.4)
    plt.plot(smooth(rewards), label="Smoothed Reward", color='dodgerblue')
    plt.title("Episode Rewards Over Time", fontsize=14)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(alpha=0.3)

    # Loss curve
    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Mini-Batch Loss", color='darkorange')
    plt.title("Mini-Batch Loss During Training", fontsize=14)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("vanilla_dqn_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved training plot as vanilla_dqn_results.png")

    # ===============================================================
    # Evaluate the trained agent
    # ===============================================================
    test_episodes = 100
    scores = []
    for _ in range(test_episodes):
        s, _ = env.reset(seed=SEED)
        done, truncated, ep_r = False, False, 0
        while not done and not truncated:
            s_t = torch.FloatTensor(s).unsqueeze(0)
            # Evaluate with no exploration (epsilon=0)
            a = epsilon_greedy(model, s_t, epsilon=0.0, action_space=env.action_space.n)
            s, r, done, truncated, _ = env.step(a)
            ep_r += r
        scores.append(ep_r)
    print(f"Average Test Reward over {test_episodes} episodes: {np.mean(scores):.2f}")

if __name__ == "__main__":
    main()