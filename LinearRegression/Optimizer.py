from LinearRegressor import LinearRegressor
from LossFunction import LossFunction
import numpy as np


class Optimizer:
    def __init__(self, model: LinearRegressor, loss_func: LossFunction, learning_rate=0.01):
        self.model = model
        self.loss_func = loss_func
        self.learning_rate = learning_rate

    def train(self,X: np.ndarray, y: np.ndarray):
        for i in range(1000):
            for i in range(X.shape[1]):
                self.model.forward(X[:,i])
                self.loss_func.forward(y[:,i], self.model.output)
                self.loss_func.grad()
                self.model.grad()
            self.loss_func.gradient_y_pred = self.loss_func.gradient_y_pred/X.shape[1]
            self.model.backward(self.loss_func.gradient_y_pred, self.learning_rate, X.shape[1])
            self.loss_func.gradient_y_pred = None