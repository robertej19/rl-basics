import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Use CUDA if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

print("Using device:", device)

# --- Simulation Function ---
def simulation(launch_angle: float) -> float:
    gravity = 9.8
    velocity = 5
    launch_angle = np.radians(launch_angle)
    distance = (velocity**2 * np.sin(2 * launch_angle)) / gravity
    return distance

# --- Environment ---
class BallLaunchEnv:
    def __init__(self, low=0.0, high=90.0):
        self.low = low
        self.high = high

    def reset(self):
        # Randomly sample a launch angle in degrees between low and high
        self.launch_angle = random.uniform(self.low, self.high)
        return np.array([self.launch_angle], dtype=np.float32)

    def step(self, action):
        # Compute the true distance using the simulation function
        true_distance = simulation(self.launch_angle)
        # Reward: negative squared error between predicted and true distance
        reward = - (action[0] - true_distance) ** 2
        done = True  # one-step episode
        # New state: new random launch angle
        next_state = np.array([random.uniform(self.low, self.high)], dtype=np.float32)
        return next_state, reward, done, {}

# --- Replay Buffer ---
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

# --- Actor Network ---
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        # No final activation; we are predicting a continuous distance value.
        return self.out(x)

# --- Critic Network ---
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # Concatenated input of state (launch angle) and action (predicted distance)
        self.fc1 = nn.Linear(1 + 1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# --- Ornstein-Uhlenbeck Noise for Exploration ---
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

# --- Soft Update Function ---
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# --- Hyperparameters ---
num_episodes = 20000
batch_size = 256
gamma = 0.99
tau = 0.005
actor_lr = 1e-3
critic_lr = 1e-3

# --- Initialize Environment, Networks, Optimizers, Buffer, and Noise ---
env = BallLaunchEnv()

actor = Actor().to(device)
actor_target = Actor().to(device)
actor_target.load_state_dict(actor.state_dict())

critic = Critic().to(device)
critic_target = Critic().to(device)
critic_target.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

replay_buffer = ReplayBuffer(10000)
noise = OUNoise(size=(1,))

# --- Training Loop ---
for episode in range(num_episodes):
    state = env.reset()  # launch angle in degrees
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # shape: (1, 1)
    noise.reset()
    
    # Actor selects action with added exploration noise
    actor.eval()
    with torch.no_grad():
        action = actor(state_tensor).cpu().data.numpy().flatten()
    actor.train()
    action = action + noise.sample()  # exploration

    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    # Update networks if enough samples are available
    if len(replay_buffer) > batch_size:
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(device)       # shape: (batch, 1)
        actions = torch.FloatTensor(actions).to(device)       # shape: (batch, 1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)  # shape: (batch, 1)
        next_states = torch.FloatTensor(next_states).to(device)  # shape: (batch, 1)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)  # shape: (batch, 1)
        
        # --- Critic Update ---
        with torch.no_grad():
            next_actions = actor_target(next_states)
            target_Q = critic_target(next_states, next_actions)
            target_value = rewards + gamma * (1 - dones) * target_Q
        
        current_Q = critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_value.detach())
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        # --- Actor Update ---
        actor_loss = -critic(states, actor(states)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        # --- Soft Update Target Networks ---
        soft_update(actor_target, actor, tau)
        soft_update(critic_target, critic, tau)
    
    if episode % 500 == 0 and len(replay_buffer) > batch_size:
        print(f"Episode {episode}: reward = {reward:.4f}, actor_loss = {actor_loss.item():.4f}, critic_loss = {critic_loss.item():.4f}")

# --- Evaluation and Visualization ---
# Evaluate the learned mapping across launch angles from 0 to 90 degrees.
angles = np.linspace(0, 90, 100)
angles_tensor = torch.FloatTensor(angles).unsqueeze(1).to(device)
with torch.no_grad():
    predicted = actor(angles_tensor).squeeze().cpu().numpy()

true_distances = np.array([simulation(angle) for angle in angles])

plt.figure(figsize=(8, 5))
plt.plot(angles, predicted, label="Predicted Distance")
plt.plot(angles, true_distances, label="True Distance", linestyle="dashed")
plt.xlabel("Launch Angle (degrees)")
plt.ylabel("Distance")
plt.title("DDPG: Learning Ball Launch Distance")
plt.legend()
plt.show()
