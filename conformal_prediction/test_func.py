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
    x = np.linspace(-2*np.pi, 2*np.pi, 200)
    y = true_function(x)
    data = np.column_stack((x, y))
    np.random.shuffle(data)
    
    # Split the data into training and testing sets
    split_idx = int(0.8 * len(data))
    train_data, test_data = data[:split_idx], data[split_idx:]

    x_train = torch.tensor(train_data[:, 0], dtype=torch.float32).view(-1, 1)
    y_train = torch.tensor(train_data[:, 1], dtype=torch.float32).view(-1, 1)
    x_test = torch.tensor(test_data[:, 0], dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(test_data[:, 1], dtype=torch.float32).view(-1, 1)

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
        residuals = np.abs(y_train - predictions)
    
    residuals = residuals.flatten()
    residuals_sorted = np.sort(residuals)

    # Determine the error bounds at a given confidence level (95%)
    delta = 0.05
    quantile = np.quantile(residuals_sorted, 1 - delta,method='higher')
    C = residuals_sorted[np.ceil(( (len(x_train))*(1-delta) )).astype(int)]
    print("R(k): ", np.ceil(( (len(x_train)) * (1-delta))).astype(int))
    print("Quantile: ", quantile)
    print("C: ", C)


    # Prediction with bounds
    model.eval()
    with torch.no_grad():
        predictions = model(x_test).numpy().flatten()
        x_test = x_test.numpy().flatten()
        error = C*np.ones_like(predictions)

    plt.plot(x, y, label='True function', color='red')
    plt.errorbar(x_test, predictions, yerr=error, fmt='o', label='NN approximation with error bars')
    plt.legend()
    plt.show()
