# DISRC_ASSIP

DISRC_ASSIP is a repository containing two separate implementations of reinforcement learning agents trained on the CartPole‑v1 environment. The first implementation is a standard Vanilla DQN (Deep Q‑Network) that uses a fully connected network, experience replay, epsilon‑greedy exploration, and a target network. The second implementation is a DISRC DQN (Deep Intrinsic Surprise‑Regularized Control), which builds on the standard DQN by introducing a biologically inspired surprise‑based mechanism that shapes rewards according to deviations in latent state encodings.

Both agents are trained and evaluated under identical hyperparameters and random seeds to ensure fair comparison. Each script trains the agent, plots and saves reward and loss curves, and evaluates performance over a fixed number of test episodes. This repository is intended to provide a clear baseline and an enhanced variant for research on stability, efficiency, and learning performance in deep reinforcement learning.

Authors:
- Yash Kini (kiniyash3@gmail.com)

- Shiv Davay (davayshiv@gmail.com)

- Shreya Polavarapu (shreyapolavarapu9@gmail.com)

Please contact the authors for any questions, issues, or collaboration requests regarding this repository.

Developed during ASSIP (Aspiring Scientists Summer Internship Program) 2025.

