"""
Unsupervised Machine Learning Hybrid Approach Integrating Linear Programming

This script implements a hybrid approach that combines linear programming (LP) with an unsupervised machine learning model.
The LP constraints are integrated directly into the loss function of an autoencoder, allowing the model to learn meaningful
representations of the data while adhering to domain-specific optimization constraints.

Model Architecture:
- Encoder: 
    - Linear layer with 256 units, ReLU activation
    - Linear layer with latent_size units, ReLU activation
- Latent Space:
    - Bottleneck layer with latent_size units
- Decoder:
    - Linear layer with 256 units, ReLU activation
    - Linear layer with input_size units (output layer)

Main Classes and Functions:
- Autoencoder: Defines the encoder-decoder architecture.
- LPLoss: Computes the linear programming-based loss component.
- CombinedLoss: Combines reconstruction loss and LP loss.
- generate_synthetic_data: Generates synthetic data representing resource allocation scenarios.
- train_model: Trains the autoencoder with the combined loss function.

Parameter List:
- input_size: Number of input features (6 in this example).
- latent_size: Size of the latent space (3 in this example).
- num_samples: Number of synthetic data samples to generate.
- batch_size: Batch size used for training (64 in this example).
- num_epochs: Number of training epochs (100 in this example).
- learning_rate: Learning rate for the optimizer (0.0001 in this example).
- c, A, b: Coefficients and constraints for the LP component.

Usage:
- Run the script to train the model and predict resource allocation using synthetic data.
- Example usage with new data is included in the `if __name__ == "__main__":` block.
- customize by training on your data and inference on new data for your application domain.

Andrew Kiruluta, 08/18/2024

"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_size),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(True),
            nn.Linear(256, input_size)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class LPLoss(nn.Module):
    def __init__(self, c, A, b):
        super(LPLoss, self).__init__()
        self.c = torch.tensor(c, dtype=torch.float32)
        self.A = torch.tensor(A, dtype=torch.float32)
        self.b = torch.tensor(b, dtype=torch.float32)

    def forward(self, predictions, latent):
        latent = torch.relu(latent)
        lp_obj = torch.matmul(latent, self.c)
        constraint_violation = torch.relu(torch.matmul(latent, self.A.T) - self.b)
        penalty = torch.sum(constraint_violation, dim=1)
        loss = torch.mean(lp_obj + penalty)
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, lp_loss, unsupervised_loss_weight=1.0, lp_loss_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.lp_loss = lp_loss
        self.unsupervised_loss_weight = unsupervised_loss_weight
        self.lp_loss_weight = lp_loss_weight

    def forward(self, predictions, inputs, latent):
        reconstruction_loss = nn.MSELoss()(predictions, inputs)
        lp_loss_value = self.lp_loss(predictions, latent)
        combined_loss = (self.unsupervised_loss_weight * reconstruction_loss +
                         self.lp_loss_weight * lp_loss_value)
        return combined_loss

# replace with custom data specific to your needs
def generate_synthetic_data(num_samples=1000):
    num_features = 6
    available_doctors = torch.randint(1, 11, (num_samples, 1)).float()
    available_nurses = torch.randint(1, 21, (num_samples, 1)).float()
    available_equipment = torch.randint(1, 16, (num_samples, 1)).float()
    max_time_available = torch.rand(num_samples, 1) * 15 + 5
    treatment_1_time_requirement = torch.rand(num_samples, 1) * 4 + 1
    treatment_2_time_requirement = torch.rand(num_samples, 1) * 4 + 1
    X = torch.cat([available_doctors, available_nurses, available_equipment,
                   max_time_available, treatment_1_time_requirement, treatment_2_time_requirement], dim=1)
    return X

def train_model():
    X = generate_synthetic_data(num_samples=10000)
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    input_size = 6
    latent_size = 3
    model = Autoencoder(input_size, latent_size)

    c = torch.tensor([1.0, 1.0, 1.0])
    A = torch.tensor([[1, 2, 1], [2, 1, 1], [1, 1, 2]])
    b = torch.tensor([10.0, 20.0, 15.0])
    lp_loss_function = LPLoss(c, A, b)

    combined_loss_function = CombinedLoss(lp_loss_function)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_x in dataloader:
            batch_x = batch_x[0]
            reconstructed, latent = model(batch_x)
            loss = combined_loss_function(reconstructed, batch_x, latent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    return model

if __name__ == "__main__":
    trained_model = train_model()

    # Example inference with new sample data
    new_input = torch.tensor([[5, 10, 8, 15, 2, 3],
                              [6, 12, 7, 14, 3, 2],
                              [8, 15, 5, 18, 1.5, 2.5]], dtype=torch.float32)
    trained_model.eval()
    with torch.no_grad():
        reconstructed, latent = trained_model(new_input)
        predicted_allocations = torch.relu(reconstructed)
        rounded_allocations = torch.round(predicted_allocations)

    print("Rounded Predicted Treatment Allocations:")
    print(rounded_allocations)
