import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#write a function that calculates the distance a projectile will travel given the variables above


import torch
import torch.nn as nn
import torch.optim as optim

# Use CUDA if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

print("Using device:", device)
def simulation(context: float) -> float:
    """
    Simulation returns the square of the context.
    Input: context in [0, 1]
    Output: context^2, which is also in [0, 1]
    """
    return context ** 2

# ----- Prediction Network -----
class PredictionNetwork(nn.Module):
    def __init__(self):
        super(PredictionNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 16)  # Input layer to hidden layer
        self.fc2 = nn.Linear(16, 16) # Hidden layer
        self.fc_out = nn.Linear(16, 1)  # Output layer
        
        # We use a sigmoid activation on the output to constrain it to [0,1]
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc_out(x)
        # Constrain the prediction to [0,1]
        return self.sigmoid(x)

# ----- Setup -----
model = PredictionNetwork().to(device)
criterion = nn.MSELoss()  # Mean squared error loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 1000
batch_size = 32

# ----- Training Loop -----
for epoch in range(1, num_epochs + 1):
    # Generate a batch of training data: uniformly sample from [0,1]
    contexts = torch.rand(batch_size, 1, device=device)  # shape: [batch_size, 1]
    targets = contexts ** 2  # true output from the simulation function
    
    # Forward pass: predict the squared values
    predictions = model(contexts)
    
    # Compute loss: MSE between predictions and true outcomes
    loss = criterion(predictions, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        # Compute average loss over this batch for logging
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

# ----- Evaluation -----
# Evaluate on a set of test values from 0 to 1.
model.eval()
with torch.no_grad():
    test_contexts = torch.linspace(0, 1, steps=10).unsqueeze(1).to(device)
    test_predictions = model(test_contexts)
    true_values = test_contexts ** 2
    print("\nEvaluation:")
    for c, pred, true in zip(test_contexts, test_predictions, true_values):
        print(f"Input: {c.item():.3f} | Prediction: {pred.item():.3f} | True: {true.item():.3f}")