import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        
        dZ2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, self.W2.T) * (self.a1 * (1 - self.a1))
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate, l2_lambda=0):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            loss = binary_cross_entropy(y, output) + l2_lambda * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        
        return losses

def plot_activation_functions():
    x = np.linspace(-10, 10, 100)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, sigmoid(x))
    plt.title('Sigmoid Function')
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')

    plt.subplot(1, 2, 2)
    plt.plot(x, relu(x))
    plt.title('ReLU Function')
    plt.xlabel('x')
    plt.ylabel('ReLU(x)')

    plt.tight_layout()
    plt.show()
    plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_regularization_comparison(losses, losses_regularized):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Without Regularization')
    plt.plot(losses_regularized, label='With L2 Regularization')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig('regularization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_activation_functions()

    # XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Train without regularization
    nn = NeuralNetwork(2, 4, 1)
    losses = nn.train(X, y, epochs=10000, learning_rate=0.1)
    plot_training_loss(losses)

    # Train with L2 regularization
    nn_regularized = NeuralNetwork(2, 4, 1)
    losses_regularized = nn_regularized.train(X, y, epochs=10000, learning_rate=0.1, l2_lambda=0.01)
    
    plot_regularization_comparison(losses, losses_regularized)

    print("All visualizations have been saved as PNG files.")