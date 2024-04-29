import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the function to approximate
def true_function(x):
    return np.cos(x)

def train_nn(x_train,y_train,epochs=1000):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(x_train)  # Forward pass
        loss = criterion(outputs, y_train)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        if epoch % 100 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    return model


# Define a neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 100)  # Input layer to hidden layer with 50 neurons
        self.fc2 = nn.Linear(100, 1)  # Hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function for hidden layer
        x = self.fc2(x)  # Output layer
        return x


if __name__ == '__main__':
    # Create a dataset
    x = np.linspace(-2*np.pi, 2*np.pi, 200) + np.random.normal(0, 0.2, 200)
    y = true_function(x)
    data = np.column_stack((x, y))
    # np.random.shuffle(data)
    
    # Split the data into training and testing sets
    split_idx = int(0.8 * len(data))
    train_data, test_data = data[:split_idx], data[split_idx:]

    x_train = torch.tensor(train_data[:, 0], dtype=torch.float32).view(-1, 1)
    y_train = torch.tensor(train_data[:, 1], dtype=torch.float32).view(-1, 1)
    x_test = torch.tensor(test_data[:, 0], dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(test_data[:, 1], dtype=torch.float32).view(-1, 1)

    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Instantiate the model, loss function, and optimizer
    model = Net()
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

    # Training the model
    model = train_nn(x_train,y_train,epochs=2000)

    # Calibrate phase to determine residuals
    model.eval()
    with torch.no_grad():
        predictions = model(x_train)
        residuals = y_train - predictions

    # Determine the error bounds at a given confidence level (95%)
    alpha = 0.05
    quantile = np.quantile(np.abs(residuals), 1 - alpha / 2)

    # Prediction with bounds
    model.eval()
    with torch.no_grad():
        predictions = model(x_train).numpy()
        lower_bound = predictions - quantile
        upper_bound = predictions + quantile

    plt.plot(x, y, label='True function', color='red')
    plt.scatter(x_train, predictions, label='NN approximation')
    plt.fill_between(x_train.numpy().flatten(), lower_bound.flatten(), upper_bound.flatten(), alpha=0.5, label='95% confidence interval')
    plt.legend()
    plt.show()
