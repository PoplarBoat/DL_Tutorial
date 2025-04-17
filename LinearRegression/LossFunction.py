import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error Loss Function
    """
    return (y_true - y_pred)**2


funcDict = {'mse': mse}


class LossFunction:
    def __init__(self, loss_function:str):
        self.loss_function = funcDict[loss_function]
        self.gradient_y_pred = None
        self.loss = None
        self.y_true = None
        self.y_pred = None

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Forward pass of the loss function
        """
        if self.gradient_y_pred is None:
            self.gradient_y_pred=np.zeros(y_pred.shape)
        self.y_true = y_true
        self.y_pred = y_pred
        self.loss = self.loss_function(y_true, y_pred)
        return self.loss

    def grad(self):
        """
        Backward pass of the loss function
        """

        self.gradient_y_pred += 2 * (self.y_pred - self.y_true)

if __name__ == '__main__':
    loss = LossFunction('mse')
    y_true = np.array([[1], [2], [3]])
    y_pred = np.array([[2], [3], [4]])
    loss.forward(y_true, y_pred)
    loss.grad()
    print(loss.gradient_y_pred)