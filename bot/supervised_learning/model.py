import numpy as np
import os

model_file = 'saved_model/spv_numpy_model.npz'

class Model:

    def __init__(self, input_layer: int = 12, hidden_layer: int = 256, output_layer: int = 9, learning_rate: float = 0.01):

        self.learning_rate = learning_rate
        self.model_path = model_file
        
        if os.path.exists(self.model_path):
            self.load()
        else:
            # generate random weight and bias with all element between -0.5 and 0.5
            self.__random_weight_and_bias(input_layer, hidden_layer, output_layer)

    def __random_weight_and_bias(self, input_layer: int, hidden_layer: int, output_layer: int) -> None:
        self.weight_1       = np.random.rand(hidden_layer, input_layer) - 0.5
        self.bias_1         = np.random.rand(hidden_layer, 1) - 0.5
        self.weight_2       = np.random.rand(output_layer, hidden_layer) - 0.5
        self.bias_2         = np.random.rand(output_layer, 1) - 0.5

    def forward(self, input: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # calculate values after every layer
        raw_hidden_output   = self.weight_1.dot(input) + self.bias_1
        act_hidden_output   = self.__ReLU(raw_hidden_output)
        raw_output          = self.weight_2.dot(act_hidden_output) + self.bias_2
        raw_output          = self.__soft_max(raw_output)
        return raw_hidden_output, act_hidden_output, raw_output
    
    def __backpropagation(self, model_raw_hidden_output: np.ndarray, model_act_hidden_output: np.ndarray, model_raw_output: np.ndarray, input: np.ndarray, expected_output: np.ndarray) -> None:
        # calculate deltas
        loss                = model_raw_output - expected_output
        delta_weight_2      = 1 / loss.size * loss.dot(model_act_hidden_output.T)
        delta_bias_2        = (1 / loss.size) * np.sum(loss, axis=1, keepdims=True)
        delta_hidden        = self.weight_2.T.dot(loss) * self.__derivative_ReLU(model_raw_hidden_output)
        delta_weight_1      = 1 / delta_hidden.size * delta_hidden.dot(input.T)
        delta_bias_1        = (1 / delta_hidden.size) * np.sum(delta_hidden, axis=1, keepdims=True)

        # update weight and bias
        self.weight_1       = self.weight_1 - self.learning_rate * delta_weight_1
        self.bias_1         = self.bias_1 - self.learning_rate * delta_bias_1
        self.weight_2       = self.weight_2 - self.learning_rate * delta_weight_2
        self.bias_2         = self.bias_2 - self.learning_rate * delta_bias_2

    def __ReLU(self, A: np.ndarray) -> np.ndarray:
        return np.maximum(0, A)
    
    def __derivative_ReLU(self, weight: np.ndarray) -> np.ndarray:
        return weight > 0
    
    def __soft_max(self, A: np.ndarray) -> np.ndarray:
        exp_A = np.exp(A - np.max(A))  # improve numerical stability
        soft_max = exp_A / np.sum(exp_A, axis=0, keepdims=True)
        return soft_max
    
    def train(self, input: np.ndarray, expected_output: np.ndarray):
        # train a single data / train short memory
        raw_hidden_output, act_hidden_output, raw_output = self.forward(input)
        self.__backpropagation(raw_hidden_output, act_hidden_output, raw_output, input, expected_output)
    
    def set_model_path(self, model_path: str) -> None:
        self.model_path = model_path
    
    def save(self) -> None:
        np.savez(self.model_path,
            weight_1=self.weight_1,
            bias_1=self.bias_1,
            weight_2=self.weight_2,
            bias_2=self.bias_2)
    
    def load(self) -> None:
        data = np.load(self.model_path)
        
        # Gán lại trọng số và bias
        self.weight_1 = data["weight_1"]
        self.bias_1 = data["bias_1"]
        self.weight_2 = data["weight_2"]
        self.bias_2 = data["bias_2"]