"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        optimizer
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.outputs = {}
        self.gradients = {}
        self.m = {}
        self.v = {}
        self.x = 0

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])

            self.params["b" + str(i)] = np.zeros(sizes[i])

            self.m[f"W{i}"] = np.zeros((self.params[f"W{i}"]).shape)
            self.m[f"b{i}"] = np.zeros((self.params[f"b{i}"]).shape)

            self.v[f"W{i}"] = np.zeros((self.params[f"W{i}"]).shape)
            self.v[f"b{i}"] = np.zeros((self.params[f"b{i}"]).shape)

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        return np.dot(X, W) + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(X, 0)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        grad = X.copy()
        grad[grad <= 0] = 0
        grad[grad > 0] = 1
        return grad

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
      # TODO ensure that this is numerically stable
      #reference: https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
      return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

    def sigmoid_grad(self, x: np.ndarray) -> np.ndarray:
      return self.sigmoid(x) * (1 - self.sigmoid(x))

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
      # TODO implement this
      return np.mean((y - p) ** 2)

    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
      return -2 * (y - p) / y.shape[0] / y.shape[1]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}

        A = X
        # Loop through each layer
        output_layer = self.num_layers
        final = output_layer + 1
        for i in range(1, final):
            # Get the index of the current layer
            current_index = i - 1
            temp = A

            # Compute linear_output using the linear function and current layer params
            linear_output = self.linear(self.params[f"W{i}"], temp, self.params[f"b{i}"])
            
            # Compute relu_output 
            if i == output_layer:
                # Use sigmoid activation for the output layer
                A = self.sigmoid(linear_output)
            else:
                # Use ReLU activation for hidden layers
                A = self.relu(linear_output)
                
            # Save the outputs for this layer
            self.outputs[f"Z{i}"] = linear_output
            self.outputs[f"A{current_index}"] = temp

            # Save the final output and return it
        self.outputs[f"A{output_layer}"] = A
        return A

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        
        output_layer = self.num_layers
        prediction = self.outputs[f"A{output_layer}"]
        loss_temp = self.mse_grad(y, prediction) * self.sigmoid_grad(self.outputs[f"Z{output_layer}"])

        # Compute the gradients for the output layer
        temp = self.outputs[f"A{output_layer - 1}"].T
        self.gradients[f"W{output_layer}"] = np.dot(temp,loss_temp)
        self.gradients[f"b{output_layer}"] = np.sum(loss_temp, axis=0)

        # Compute the gradients for the hidden layers
        for layer in range(output_layer - 1, 0, -1):
            # Compute the loss gradient for the current layer
            relu_grad = self.relu_grad(self.outputs[f"Z{layer}"])
            loss_temp = (loss_temp @ self.params[f"W{layer+1}"].T) * relu_grad

            # Compute the gradients for the current layer
            prev = self.outputs[f"A{layer-1}"]
            self.gradients[f"W{layer}"] = np.dot(prev.T,loss_temp)
            self.gradients[f"b{layer}"] = np.sum(loss_temp, axis=0)

        # Compute and return the loss
        loss = self.mse(y, prediction)
        return loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        if opt == "Adam":
            self.x += 1
            for i in self.params:
                self.m[i] = b1 * self.m[i] + (1 - b1) * self.gradients[i]
                self.v[i] = b2 * self.v[i] + (1 - b2) * (self.gradients[i]**2)
                m_hat = (b1 * self.m[i] + (1 - b1) * self.gradients[i]) / (1 - np.power(b1, self.x))
                v_hat = (b2 * self.v[i] + (1 - b2) * (self.gradients[i]**2)) / (1 - np.power(b2, self.x))
                self.params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        elif opt == "SGD":
            for i in self.params:
                self.params[i] -= lr * self.gradients[i]