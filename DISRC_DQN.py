# ================================================================
# DISRC DQN (Deep Intrinsic Surprise‑Regularized Control)
# ================================================================
# This script trains a DQN agent augmented with a biologically-inspired
# surprise-based mechanism (DISRC) on the CartPole-v1 environment.
#
# Key Features:
#  - Encoder network to map raw observations to latent features
#  - Surprise-based bonus to scale rewards depending on TD-error-like novelty
#  - LayerNorm and gradient clipping for stability
#  - Target network with soft updates
#  - Replay buffer for experience replay
#
# Authors: Yash Kini, Shiv Davay, Shreya Polavarapu
# ================================================================

import gymnasium as gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# ================================================================
# Reproducibility
# ================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ================================================================
# Utility Functions
# ================================================================
def ensure_2d_tensor(tensor):
    """
    Ensure the input is a 2D torch tensor (batch x features).
    Converts numpy arrays to torch tensors as needed.
    """
    if isinstance(tensor, np.ndarray):
        tensor = torch.FloatTensor(tensor)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

def init_weights(m):
    """
    Initialize weights for Linear layers with Xavier uniform initialization
    and biases with zeros for stable training.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# ================================================================
# DISRC Controller
# ================================================================
class DISRCController:
    """
    Surprise-based controller:
    - Maintains a running setpoint of encoded states.
    - Computes deviation between current encoded state and setpoint.
    - Produces a bonus that decreases as training progresses.
    """
    def __init__(self, state_dim, alpha=0.03, beta_start=0.04):
        self.state_dim = state_dim
        self.alpha = alpha           # smoothing for setpoint updates
        self.beta_start = beta_start # initial surprise scaling
        self.setpoint = np.zeros(state_dim, dtype=np.float32)

    def compute_bonus(self, encoded_state, episode_ratio):
        """
        Compute surprise bonus:
        :param encoded_state: latent state vector (torch tensor)
        :param episode_ratio: current_episode / total_episodes
        """
        s_np = encoded_state.detach().cpu().numpy().squeeze()
        s_norm = s_np / (np.linalg.norm(s_np) + 1e-8)
        sp_norm = self.setpoint / (np.linalg.norm(self.setpoint) + 1e-8)

        deviation = np.linalg.norm(s_norm - sp_norm)
        beta = self.beta_start * (1.0 - episode_ratio ** 1.2)  # decay bonus
        bonus = -beta * deviation

        # update the running setpoint
        self.setpoint = (1.0 - self.alpha) * self.setpoint + self.alpha * s_np
        return bonus

# ================================================================
# Model Architectures
# ================================================================
class DISRCStateEncoder(nn.Module):
    """
    Maps raw environment states (4-dim) into a lower-dimensional latent space (4-dim).
    LayerNorm is used to stabilize learning.
    """
    def __init__(self, input_dim=4, encoded_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, encoded_dim)
        )
    def forward(self, x):
        return self.encoder(x)

class DISRC_DQN(nn.Module):
    """
    Q-network that takes encoded state features and outputs Q-values for each action.
    Uses LayerNorm and ReLU activations for stability.
    """
    def __init__(self, input_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
    def forward(self, x):
        return self.net(x)

# ================================================================
# Action Selection
# ================================================================
def epsilon_greedy(model, state, epsilon, action_space):
    """
    Epsilon-greedy policy:
    With probability epsilon, take a random action.
    Otherwise, take the action with the highest Q-value.
    """
    if random.random() < epsilon:
        return random.randrange(action_space)
    with torch.no_grad():
        q_values = model(state)
        return torch.argmax(q_values).item()

def soft_update(target, source, tau=0.005):
    """
    Soft update of target network parameters:
    θ_target = τ*θ_source + (1-τ)*θ_target
    """
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

# ================================================================
# Training Loop
# ================================================================
def train_dqn(env, model, target_model, encoder,
              encoder_optimizer, model_optimizer,
              disrc_controller,
              num_episodes=800, gamma=0.99,
              epsilon_start=1.0, epsilon_min=0.05,
              batch_size=128):
    """
    Core training loop:
    - Collect experience in replay buffer
    - Sample mini-batches for training after warm-up
    - Apply surprise bonus during reward shaping
    """
    replay_buffer = deque(maxlen=100000)
    all_rewards, losses = [], []
    epsilon = epsilon_start
    reward_norm = 1.0  # running reward normalization

    for episode in range(num_episodes):
        state_arr, _ = env.reset(seed=SEED)
        state_tensor = torch.FloatTensor(state_arr).unsqueeze(0)
        done, total_reward = False, 0

        while not done:
            # encode state and pick action
            encoded_state = encoder(state_tensor)
            action = epsilon_greedy(model, encoded_state, epsilon, env.action_space.n)

            # step in environment
            next_state_arr, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # compute shaped reward
            reward_norm = 0.99 * reward_norm + 0.01 * abs(reward)
            normed_reward = reward / (reward_norm + 1e-8)
            bonus = disrc_controller.compute_bonus(encoded_state, episode / num_episodes)
            shaped_reward = np.clip(normed_reward + 0.2 * bonus, -1.0, 1.0)

            # store transition
            next_tensor = torch.FloatTensor(next_state_arr).unsqueeze(0)
            replay_buffer.append((state_tensor.detach(), action, shaped_reward,
                                  next_tensor.detach(), float(done)))
            state_tensor = next_tensor

            # learn from replay buffer if enough samples
            if len(replay_buffer) >= 5000:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards_b, next_states, dones = zip(*batch)
                states = torch.cat([ensure_2d_tensor(s) for s in states])
                next_states = torch.cat([ensure_2d_tensor(ns) for ns in next_states])
                actions = torch.LongTensor(actions)
                rewards_b = torch.FloatTensor(rewards_b)
                dones = torch.FloatTensor(dones)

                # encode states
                s_enc = encoder(states)
                ns_enc = encoder(next_states)

                # compute targets
                with torch.no_grad():
                    next_q = model(ns_enc)
                    next_act = torch.argmax(next_q, dim=1)
                    next_q_target = target_model(ns_enc)
                    target_vals = next_q_target.gather(1, next_act.view(-1, 1)).squeeze(1)
                    targets = rewards_b + gamma * target_vals * (1 - dones)

                q_vals = model(s_enc).gather(1, actions.view(-1, 1)).squeeze(1)
                loss = (q_vals - targets).pow(2).mean()

                # backpropagation
                model_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
                model_optimizer.step()
                encoder_optimizer.step()
                losses.append(loss.item())

                # update target network
                soft_update(target_model, model, tau=0.005)

        # record reward and decay epsilon
        all_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * 0.995)

        if episode % 10 == 0:
            mean_reward = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)
            mean_loss = np.mean(losses[-100:]) if losses else 0.0
            print(f"[Ep {episode:03d}] Reward: {total_reward:.2f} | "
                  f"Mean(50): {mean_reward:.2f} | "
                  f"Eps: {epsilon:.3f} | Loss: {mean_loss:.4f}")

    return all_rewards, losses

# ================================================================
# Main entry point
# ================================================================
def main():
    # create environment
    env = gym.make("CartPole-v1")

    # build encoder and Q-networks
    encoder = DISRCStateEncoder(input_dim=4, encoded_dim=4)
    model = DISRC_DQN(input_size=4, action_size=env.action_space.n)
    target_model = DISRC_DQN(input_size=4, action_size=env.action_space.n)
    target_model.load_state_dict(model.state_dict())

    # weight initialization
    encoder.apply(init_weights)
    model.apply(init_weights)
    target_model.apply(init_weights)

    # optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=3e-4)
    model_optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # controller for surprise-based reward shaping
    disrc_controller = DISRCController(state_dim=4, alpha=0.03, beta_start=0.04)

    # train agent
    rewards, losses = train_dqn(env, model, target_model, encoder,
                                encoder_optimizer, model_optimizer,
                                disrc_controller)

    # helper function for smoothing
    def smooth(data, w=0.9):
        sm, last = [], data[0]
        for x in data:
            last = last * w + (1 - w) * x
            sm.append(last)
        return sm

    # plot results and save as PNG
    plt.figure(figsize=(12, 5))

    # plot reward curve
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Raw Reward", color='royalblue', alpha=0.4)
    plt.plot(smooth(rewards), label="Smoothed Reward", color='dodgerblue')
    plt.title("Episode Rewards Over Time", fontsize=14)
    plt.xlabel("Episode"); plt.ylabel("Total Reward")
    plt.legend(); plt.grid(alpha=0.3)

    # plot loss curve
    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Mini-Batch Loss", color='darkorange')
    plt.title("Mini-Batch Loss During Training", fontsize=14)
    plt.xlabel("Training Step"); plt.ylabel("Loss")
    plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout()
    # save the figure
    plt.savefig("DISRC_training_results.png")
    plt.show()
    print("Plot saved as 'DISRC_training_results.png'")
    
    # Evaluate trained agent
    test_episodes = 100
    scores = []
    for _ in range(test_episodes):
        s, _ = env.reset(seed=SEED)
        done, truncated, ep_r = False, False, 0
        while not done and not truncated:
            s_t = torch.FloatTensor(s).unsqueeze(0)
            # epsilon=0.0 means pure exploitation
            a = epsilon_greedy(model, s_t, epsilon=0.0, action_space=env.action_space.n)
            s, r, done, truncated, _ = env.step(a)
            ep_r += r
        scores.append(ep_r)
    print(f"Average Test Reward over {test_episodes} episodes: {np.mean(scores):.2f}")


if __name__ == "__main__":
    main()
