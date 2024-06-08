import numpy as np

class LogisticRegressionModel:
    def __init__(self):
        self.params = None
    
    def fit(self, x: np.ndarray, y: np.ndarray, steps: int, learning_rate: float, debug: bool) -> None:
        """
        Args:
            x: features - batch x n
            y: targets - batch x 1 (0s and 1s)
            steps: training steps
            debug: print training loss at the end of every training step
        """
        if self.params:
            raise RuntimeError('Model has already been trained.')
        
        # number of data points
        m = x.shape[0]
        
        # init parameters
        params = self._init_params(x)
        print(f'params: {params}')
        # training loop
        for i in range(steps):
            # calculate sigmoid function
            sigmoid = self._sigmoid(x, params) # 1 x batch
            #print(f'sigmoid: {sigmoid}')
            # calculate gradient
            delta = ((sigmoid - y.T) @ x) / m
            #print(f'delta: {delta}')
            # grandient decent
            params = params - learning_rate * delta
            
            # calculate loss
            loss = (-np.log(sigmoid) @ y - np.log(1 - sigmoid) @ (1 - y)) / m
            if debug and (i+1) % 100 == 0:
                print(f'*** Training step {i+1} - loss: {loss}')
        
        self.params = params
    
    def predict(self, x: np.ndarray, threshold: float = 0.5, y: np.ndarray | None = None) -> np.ndarray:
        """
        Args:
            x: features - batch x n
            threshold: The prediction is 1 if the model output is greater or equal to the threshold
            y: labels - batch x 1 (0s and 1s)
        Returns:
            batch x 1 (0s and 1s)
        """
        prediction = self._sigmoid(x, self.params).T > threshold
        if y is not None:
            accuracy = np.sum(prediction == y) / y.shape[0]
            print(f"accuracy is {accuracy}")
        return prediction
    
    def _sigmoid(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Args:
            x: features - batch x n
            y: targets - batch x 1 (0s and 1s)
        Returns:
            batch x 1 (0s and 1s)
        """
        z = (-params @ x.T)
        return 1.0 / (1.0 + np.exp(z))
        
    # TODO: there are probably better ways to initialize parameters
    def _init_params(self, x: np.ndarray) -> np.ndarray:
        """
        Returns:
            1 x n ndarray
        """
        params = np.zeros((1, x.shape[1]))
        return params


class LinearRegressionModel:
    pass
