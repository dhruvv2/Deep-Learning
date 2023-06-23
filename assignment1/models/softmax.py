"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me
        gradients = np.zeros_like(self.w) # same shape as weights
        
        N = X_train.shape[0]
        
        for i in range (N):
            xi = X_train[i]
            yi = y_train[i]
            
            mapping = self.w @ xi 
            logk = -np.max(mapping) #overflow trick
            response = mapping + logk
            denominator = np.sum(np.exp(response/self.reg_const))
            
            #cross entropy
            for classes in range(self.n_class):
                if classes == yi:
                    gradients[classes] += self.lr * (1- (np.exp(response[classes])/denominator)) * xi
                else:
                    gradients[classes] -= self.lr * (np.exp(response[classes])/denominator) * xi
            
            
        return gradients

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        
        N, D = X_train.shape
        batch_size = 512
        self.w = np.random.randn(self.n_class, D)
        
        for i in range(self.epochs):
            if i % 10 == 0:
                self.lr *= 0.9
                
            test_rows = np.random.permutation(N)[:batch_size]
            
            self.w += self.calc_gradient(X_train[test_rows], y_train[test_rows])

        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        
        predictions = (X_test @ self.w.T).argmax(axis=1)

        return predictions
