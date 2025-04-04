import numpy as np
import os

class Model:

    def __init__(self, input_layer: int = 12, hidden_layer: int = 256, output_layer: int = 9, learning_rate: float = 0.01):

        self.learning_rate = learning_rate

        if os.path.exists('weight_1.bin') and os.path.exists('bias_1.bin') and os.path.exists('weight_2.bin') and os.path.exists('bias_2.bin'):
            # load existed weight and bias 
            self.weight_1   = np.fromfile('weight_1.bin', dtype=np.float64).reshape(hidden_layer, input_layer)
            self.bias_1     = np.fromfile('bias_1.bin', dtype=np.float64).reshape(hidden_layer, 1)
            self.weight_2   = np.fromfile('weight_2.bin', dtype=np.float64).reshape(output_layer, hidden_layer)
            self.bias_2     = np.fromfile('bias_2.bin', dtype=np.float64).reshape(output_layer, 1)

        else:
            # generate random weight and bias with all element between -0.5 and 0.5
            self._random_weight_and_bias(input_layer, hidden_layer, output_layer)

    def _random_weight_and_bias(self, input_layer: int, hidden_layer: int, output_layer: int) -> None:
        self.weight_1       = np.random.rand(hidden_layer, input_layer) - 0.5
        self.bias_1         = np.random.rand(hidden_layer, 1) - 0.5
        self.weight_2       = np.random.rand(output_layer, hidden_layer) - 0.5
        self.bias_2         = np.random.rand(output_layer, 1) - 0.5

    def forward(self, input: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # calculate values after every layer
        raw_hidden_output   = self.weight_1.dot(input) + self.bias_1
        act_hidden_output   = self._ReLU(raw_hidden_output)
        raw_output          = self.weight_2.dot(act_hidden_output) + self.bias_2
        return raw_hidden_output, act_hidden_output, raw_output
    
    def _backpropagation(self, model_raw_hidden_output: np.ndarray, model_act_hidden_output: np.ndarray, model_raw_output: np.ndarray, input: np.ndarray, expected_output: np.ndarray) -> None:
        # calculate deltas
        loss                = model_raw_output - expected_output
        delta_weight_2      = 1 / loss.size * loss.dot(model_act_hidden_output.T)
        delta_bias_2        = 1 / loss.size * np.sum(loss, 2)
        delta_hidden        = self.weight_2.T.dot(loss) * self._derivative_ReLU(model_raw_hidden_output)
        delta_weight_1      = 1 / delta_hidden.size * delta_hidden.dot(input.T)
        delta_bias_1        = 1 / delta_hidden.size * np.sum(delta_hidden, 2)

        # update weight and bias
        self.weight_1       = self.weight_1 - self.learning_rate * delta_weight_1
        self.bias_1         = self.bias_1 - self.learning_rate * delta_bias_1
        self.weight_2       = self.weight_2 - self.learning_rate * delta_weight_2
        self.bias_2         = self.bias_2 - self.learning_rate * delta_bias_2

    def save(self) -> None:
        # save weight and bias
        self.weight_1.tofile('weight_1.bin')
        self.bias_1.tofile('bias_1.bin')
        self.weight_2.tofile('weight_2.bin')
        self.bias_2.tofile('bias_2.bin')

    def _ReLU(self, A: np.ndarray) -> np.ndarray:
        return np.maximum(0, A)
    
    def _derivative_ReLU(self, weight: np.ndarray) -> np.ndarray:
        return weight > 0
    
    def train(self, input: np.ndarray, expected_output: np.ndarray):
        # train a single data / train short memory
        raw_hidden_output, act_hidden_output, raw_output = self.forward(input)
        self._backpropagation(raw_hidden_output, act_hidden_output, raw_output, input, expected_output)
        