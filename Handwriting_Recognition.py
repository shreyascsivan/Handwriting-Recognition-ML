import numpy as np
from keras.datasets import mnist

# Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:1000] / 255.0  # Normalize pixel values
y_train = y_train[:1000]
x_test = x_test[:200] / 255.0  # Normalize pixel values
y_test = y_test[:200]

# Reshape the input data
x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
x_test_reshaped = x_test.reshape(x_test.shape[0], -1)

# Define the neural network architecture
input_size = 28 * 28  # flattened image size
hidden_size = 256  # Increased hidden layer size
output_size = 10  # Number of classes

# Ensure labels are integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Initialize weights and biases
np.random.seed(0)
W1 = np.random.randn(input_size, hidden_size) * 0.3  # Small random numbers close to 0
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size) * 0.3  # Small random numbers close to 0
b2 = np.zeros(output_size)

# activation fun
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Implement forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)  # Sigmoid activation for hidden layer
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return a1, z1, a2, z2

# Implement loss function (cross-entropy)
def compute_loss(y, y_pred):
    m = y.shape[0]
    return -np.sum(np.log(y_pred[np.arange(m), y] + 1e-9)) / m  # Add small epsilon for numerical stability

# Implement backward propagation with delta calculations
def backward_propagation(X, y, y_pred, learning_rate, W1, b1, W2, b2):
    m = y.shape[0]
    a1, z1, a2, z2 = forward_propagation(X, W1, b1, W2, b2)

    # Calculate the delta for the output layer
    delta_output = y_pred
    delta_output[np.arange(m), y] -= 1

    # Backpropagate the delta to the hidden layer  (sigmoid derivative)
    delta_hidden = np.dot(delta_output, W2.T) * (a1 * (1 - a1))

    # Compute gradients for weights and biases
    dW2 = np.dot(a1.T, delta_output) / m
    db2 = np.sum(delta_output, axis=0) / m
    dW1 = np.dot(X.T, delta_hidden) / m
    db1 = np.sum(delta_hidden, axis=0) / m

    # Update weights and biases
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2

# Train the neural network
num_epochs = 1000  # Increased number of epochs
learning_rate = 0.3  # Adjusted learning rate

for epoch in range(num_epochs):
    y_pred = forward_propagation(x_train_reshaped, W1, b1, W2, b2)[2]
    loss = compute_loss(y_train, y_pred)
    W1, b1, W2, b2 = backward_propagation(
        x_train_reshaped, y_train, y_pred, learning_rate, W1, b1, W2, b2)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss}')

# Evaluate the model
def predict(X, W1, b1, W2, b2):
    return np.argmax(forward_propagation(X, W1, b1, W2, b2)[2], axis=1)

predictions = predict(x_test_reshaped, W1, b1, W2, b2)
accuracy = np.mean(predictions == y_test)
print(f'Test Accuracy: {accuracy}')
