import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ---- Environment ---- #
class SquareEnv:
    def __init__(self, low=-5, high=5):
        self.low = low
        self.high = high

    def reset(self):
        # Sample a random x from [low, high]
        self.x = random.uniform(self.low, self.high)
        return np.array([self.x], dtype=np.float32)

    def step(self, action):
        # The true value is x^2; reward is negative squared error.
        true_val = self.x ** 2
        reward = - (action[0] - true_val) ** 2
        done = True  # one-step episode
        # For a new episode, sample a new random x
        next_state = np.array([random.uniform(self.low, self.high)], dtype=np.float32)
        return next_state, reward, done, {}

# ---- Replay Buffer ---- #
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# ---- Actor Network ---- #
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        # For this problem we do not squash the output
        return self.out(x)

# ---- Critic Network ---- #
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # Input will be the concatenation of state and action
        self.fc1 = nn.Linear(1 + 1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# ---- Ornstein-Uhlenbeck Noise for Exploration ---- #
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.size) * self.mu
    
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(*self.state.shape)
        self.state += dx
        return self.state

# ---- Soft Update Function ---- #
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# ---- Hyperparameters ---- #
num_episodes = 5000
batch_size = 64
gamma = 0.99
tau = 0.005
actor_lr = 1e-3
critic_lr = 1e-3

# ---- Initialize Environment, Networks, Optimizers, Buffer, and Noise ---- #
env = SquareEnv()

actor = Actor()
actor_target = Actor()
actor_target.load_state_dict(actor.state_dict())

critic = Critic()
critic_target = Critic()
critic_target.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

replay_buffer = ReplayBuffer(10000)
noise = OUNoise(size=(1,))

# ---- Training Loop ---- #
for episode in range(num_episodes):
    state = env.reset()  # state is a numpy array of shape (1,)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # shape: (1, 1)
    noise.reset()
    
    # Get action from actor with added exploration noise
    actor.eval()
    with torch.no_grad():
        action = actor(state_tensor).cpu().data.numpy().flatten()  # shape: (1,)
    actor.train()
    action = action + noise.sample()  # exploration

    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    # Update networks if enough samples are available
    if len(replay_buffer) > batch_size:
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states)       # shape: (batch, 1)
        actions = torch.FloatTensor(actions)       # shape: (batch, 1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # shape: (batch, 1)
        next_states = torch.FloatTensor(next_states)  # shape: (batch, 1)
        dones = torch.FloatTensor(dones).unsqueeze(1)  # shape: (batch, 1)

        # ---- Critic Update ---- #
        with torch.no_grad():
            next_actions = actor_target(next_states)
            target_Q = critic_target(next_states, next_actions)
            target_value = rewards + gamma * (1 - dones) * target_Q

        current_Q = critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_value.detach())

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # ---- Actor Update ---- #
        # The actor loss is the negative Q-value (to maximize Q)
        actor_loss = -critic(states, actor(states)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # ---- Soft Update the Target Networks ---- #
        soft_update(actor_target, actor, tau)
        soft_update(critic_target, critic, tau)
    
    # Print progress every 500 episodes
    if episode % 500 == 0 and len(replay_buffer) > batch_size:
        print(f"Episode {episode}: reward = {reward:.4f}, actor_loss = {actor_loss.item():.4f}, critic_loss = {critic_loss.item():.4f}")

# ---- Evaluation and Visualization ---- #
# After training, we plot the learned function versus the true function.
xs = np.linspace(-5, 5, 100)
xs_tensor = torch.FloatTensor(xs).unsqueeze(1)
with torch.no_grad():
    predicted = actor(xs_tensor).squeeze().cpu().numpy()
true = xs**2

plt.figure(figsize=(8, 5))
plt.plot(xs, predicted, label="Predicted")
plt.plot(xs, true, label="True", linestyle="dashed")
plt.xlabel("x")
plt.ylabel("x²")
plt.title("DDPG Learning the Square Function")
plt.legend()
plt.show()
