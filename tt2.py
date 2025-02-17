import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import matplotlib.pyplot as plt
import numpy as np


def simulation(launch_angle: float) -> float:
    
    gravity = 9.8
    velocity = 5
    launch_angle = np.radians(launch_angle)
    distance = (velocity**2 * np.sin(2 * launch_angle)) / gravity
    return distance

# Define a simple policy network that, given x, outputs the mean of a Gaussian.
# The network also has a learnable (state-independent) log standard deviation.
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.mean = nn.Linear(32, 1)
        # A single log_std parameter shared across all states.
        self.log_std = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # Pass the input through two tanh-activated layers.
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.mean(x)
        # Exponentiate the log_std to obtain a positive std value.
        std = self.log_std.exp()
        return mean, std

# Instantiate the policy network and the optimizer.
policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

num_episodes = 20000

for episode in range(num_episodes):
    # Sample a random input x uniformly from [-5, 5].
    x_val = torch.rand(1).item() * 80  # random value in [0,80]
    x = torch.tensor([[x_val]], dtype=torch.float32)
    true_value = simulation(x)  # our target: square of x

    # Forward pass: get the mean and standard deviation.
    mean, std = policy(x)
    # Create a normal distribution with the obtained parameters.
    dist = D.Normal(mean, std)
    # Sample an action (our network’s prediction) from the distribution.
    action = dist.sample()
    # Compute the log-probability of the taken action.
    log_prob = dist.log_prob(action)
    
    # Define the reward as the negative squared error.
    # (Maximum reward of 0 is achieved when action == x^2.)
    reward = - (action - true_value) ** 2
    # Detach the reward so that it is treated as a constant (like an environment signal).
    reward = reward.detach()
    
    # REINFORCE loss: we wish to maximize reward, so we minimize -log_prob*reward.
    loss = -log_prob * reward
    loss = loss.mean()  # in case there are extra dimensions

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 500 == 0:
        print(f"Episode {episode}: x = {x_val:.2f}, action = {action.item():.2f}, "
              f"true = {true_value.item():.2f}, reward = {reward.item():.4f}, loss = {loss.item():.4f}")

# After training, visualize the learned function.
xs = torch.linspace(0,80, 100).unsqueeze(1)
with torch.no_grad():
    predicted_mean, _ = policy(xs)
predicted = predicted_mean.squeeze().numpy()
true = (simulation(xs.squeeze().numpy()))

plt.figure(figsize=(8, 5))
plt.plot(xs.squeeze().numpy(), predicted, label='Predicted')
plt.plot(xs.squeeze().numpy(), true, label='True', linestyle='dashed')
plt.xlabel("x")
plt.ylabel("x²")
plt.title("Learning the Square Function via Reinforcement Learning")
plt.legend()
plt.show()
