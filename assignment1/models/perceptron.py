"""Perceptron model."""

import numpy as np


class Perceptron:
  def __init__(self, n_class: int, lr: float, epochs: int):
      """Initialize a new classifier.

      Parameters:
          n_class: the number of classes
          lr: the learning rate
          epochs: the number of epochs to train for
      """
      self.w = None  # TODO: change this
      self.lr = lr
      self.epochs = epochs
      self.n_class = n_class

  def train(self, X_train: np.ndarray, y_train: np.ndarray):
      """Train the classifier.

      Use the perceptron update rule as introduced in the Lecture.

      Parameters:
          X_train: a number array of shape (N, D) containing training data;
              N examples with D dimensions
          y_train: a numpy array of shape (N,) containing training labels
      """
      # TODO: implement me
      N = X_train.shape[0]
      D = X_train.shape[1]
      self.w = np.zeros((self.n_class, D))
      for epoch in range(self.epochs):
        for example in range(N):
          max_dp = 0
          predicted_class = 0
          for a_class in range(self.n_class):
              dp = np.dot(self.w[a_class].T, X_train[example])
              if dp > max_dp:
                  max_dp = dp
                  predicted_class = a_class
              
          if y_train[example] != predicted_class:
              self.w[y_train[example]] += self.lr * X_train[example]
              self.w[predicted_class] -= self.lr * X_train[example] # multi-class
                
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
      
      N = X_test.shape[0]
      pred_labels = []
      for example in range(N):
          max_dp = 0
          predicted_class = 0
          for a_class in range(self.n_class):
              dp = np.dot(self.w[a_class].T, X_test[example])
              if dp > max_dp:
                  max_dp = dp
                  predicted_class = a_class
          pred_labels.append(predicted_class)
      
      return pred_labels