import gymnasium as gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# ===============================
# Reproducibility
# ===============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===============================
# Utility
# ===============================
def ensure_2d_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.FloatTensor(tensor)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# ===============================
# DISRC Controller
# ===============================
class DISRCController:
    def __init__(self, state_dim, alpha=0.03, beta_start=0.04):
        self.state_dim = state_dim
        self.alpha = alpha
        self.beta_start = beta_start
        self.setpoint = np.zeros(state_dim, dtype=np.float32)

    def compute_bonus(self, encoded_state, episode_ratio):
        s_np = encoded_state.detach().cpu().numpy().squeeze()
        s_norm = s_np / (np.linalg.norm(s_np) + 1e-8)
        sp_norm = self.setpoint / (np.linalg.norm(self.setpoint) + 1e-8)
        deviation = np.linalg.norm(s_norm - sp_norm)
        # bonus decays gently over time
        beta = self.beta_start * (1.0 - episode_ratio ** 1.2)
        bonus = -beta * deviation
        # update setpoint
        self.setpoint = (1.0 - self.alpha) * self.setpoint + self.alpha * s_np
        return bonus

# ===============================
# Models with LayerNorm
# ===============================
class DISRCStateEncoder(nn.Module):
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

# ===============================
# Epsilon Greedy
# ===============================
def epsilon_greedy(model, state, epsilon, action_space):
    if random.random() < epsilon:
        return random.randrange(action_space)
    with torch.no_grad():
        q_values = model(state)
        return torch.argmax(q_values).item()

# ===============================
# Soft update
# ===============================
def soft_update(target, source, tau=0.005):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

# ===============================
# Training
# ===============================
def train_dqn(env, model, target_model, encoder,
              encoder_optimizer, model_optimizer,
              disrc_controller,
              num_episodes=800, gamma=0.99,
              epsilon_start=1.0, epsilon_min=0.05,
              batch_size=128):
    replay_buffer = deque(maxlen=100000)
    all_rewards, losses = [], []
    epsilon = epsilon_start
    reward_norm = 1.0

    for episode in range(num_episodes):
        state_arr, _ = env.reset(seed=SEED)
        state_tensor = torch.FloatTensor(state_arr).unsqueeze(0)
        done, total_reward = False, 0

        while not done:
            encoded_state = encoder(state_tensor)
            action = epsilon_greedy(model, encoded_state, epsilon, env.action_space.n)
            next_state_arr, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # reward shaping
            reward_norm = 0.99 * reward_norm + 0.01 * abs(reward)
            normed_reward = reward / (reward_norm + 1e-8)
            bonus = disrc_controller.compute_bonus(encoded_state, episode / num_episodes)
            shaped_reward = np.clip(normed_reward + 0.2 * bonus, -1.0, 1.0)

            next_tensor = torch.FloatTensor(next_state_arr).unsqueeze(0)
            replay_buffer.append((state_tensor.detach(), action, shaped_reward,
                                  next_tensor.detach(), float(done)))
            state_tensor = next_tensor

            if len(replay_buffer) >= 5000:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards_b, next_states, dones = zip(*batch)
                states = torch.cat([ensure_2d_tensor(s) for s in states])
                next_states = torch.cat([ensure_2d_tensor(ns) for ns in next_states])
                actions = torch.LongTensor(actions)
                rewards_b = torch.FloatTensor(rewards_b)
                dones = torch.FloatTensor(dones)

                s_enc = encoder(states)
                ns_enc = encoder(next_states)

                with torch.no_grad():
                    next_q = model(ns_enc)
                    next_act = torch.argmax(next_q, dim=1)
                    next_q_target = target_model(ns_enc)
                    target_vals = next_q_target.gather(1, next_act.view(-1, 1)).squeeze(1)
                    targets = rewards_b + gamma * target_vals * (1 - dones)

                q_vals = model(s_enc).gather(1, actions.view(-1, 1)).squeeze(1)
                loss = (q_vals - targets).pow(2).mean()

                model_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                loss.backward()
                # gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
                model_optimizer.step()
                encoder_optimizer.step()
                losses.append(loss.item())

                soft_update(target_model, model, tau=0.005)

        all_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * 0.995)

        if episode % 10 == 0:
            mean_reward = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards)
            mean_loss = np.mean(losses[-100:]) if losses else 0.0
            print(f"Episode {episode}/{num_episodes} | "
                  f"Reward: {total_reward:.2f} | MeanReward(50): {mean_reward:.2f} | "
                  f"Epsilon: {epsilon:.4f} | Loss: {mean_loss:.4f}")

    return all_rewards, losses

# ===============================
# Main
# ===============================
def main():
    env = gym.make("CartPole-v1")
    encoder = DISRCStateEncoder(input_dim=4, encoded_dim=4)
    model = DISRC_DQN(input_size=4, action_size=env.action_space.n)
    target_model = DISRC_DQN(input_size=4, action_size=env.action_space.n)
    target_model.load_state_dict(model.state_dict())

    encoder.apply(init_weights)
    model.apply(init_weights)
    target_model.apply(init_weights)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=3e-4)
    model_optimizer = optim.Adam(model.parameters(), lr=1e-4)

    disrc_controller = DISRCController(state_dim=4, alpha=0.03, beta_start=0.04)

    rewards, losses = train_dqn(env, model, target_model, encoder,
                                encoder_optimizer, model_optimizer,
                                disrc_controller)

    def smooth(data, w=0.9):
        sm, last = [], data[0]
        for x in data:
            last = last * w + (1 - w) * x
            sm.append(last)
        return sm

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Raw Reward", color='royalblue', alpha=0.4)
    plt.plot(smooth(rewards), label="Smoothed Reward", color='dodgerblue')
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Episode"); plt.ylabel("Total Reward")
    plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Mini-Batch Loss", color='darkorange')
    plt.title("Mini-Batch Loss During Training")
    plt.xlabel("Training Step"); plt.ylabel("Loss")
    plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
