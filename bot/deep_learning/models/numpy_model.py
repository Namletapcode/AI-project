import numpy as np
import os

class Model:

    def __init__(self, input_layer: int = 28, hidden_layer: int = 256, output_layer: int = 9, learning_rate: float = 0.001, discount_factor: float = 0.99, model_path: str=None, load_saved_model: bool = True):

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.model_path = model_path
        
        folder_path = os.path.dirname(self.model_path)
        os.makedirs(folder_path, exist_ok=True)
        
        if load_saved_model and os.path.exists(self.model_path):
            self.load()
        else:
            # generate random weight and bias with all element between -0.5 and 0.5
            self.__random_weight_and_bias(input_layer, hidden_layer, output_layer)

    def __random_weight_and_bias(self, input_layer: int, hidden_layer: int, output_layer: int) -> None:
        self.main_weight_1  = np.random.rand(hidden_layer, input_layer) - 0.5
        self.main_bias_1    = np.random.rand(hidden_layer, 1) - 0.5
        self.main_weight_2  = np.random.rand(output_layer, hidden_layer) - 0.5
        self.main_bias_2    = np.random.rand(output_layer, 1) - 0.5

        self.target_weight_1= self.main_weight_1.copy()
        self.target_bias_1  = self.main_bias_1.copy()
        self.target_weight_2= self.main_weight_2.copy()
        self.target_bias_2  = self.main_bias_2.copy()

    def forward(self, input: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # calculate values after every layer
        raw_hidden_output   = self.main_weight_1.dot(input) + self.main_bias_1
        act_hidden_output   = self.__ReLU(raw_hidden_output)
        raw_output          = self.main_weight_2.dot(act_hidden_output) + self.main_bias_2
        return raw_hidden_output, act_hidden_output, raw_output
    
    def predict(self, input: np.ndarray) -> np.ndarray:
        return self.forward(input)[2]
    
    def compute_target(self, current_state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, game_over: bool) -> np.ndarray:
        # use simplified Bellman equation to calculate expected output
        target = self.predict(current_state)
        if not game_over:
            Q_new = reward + self.discount_factor * np.max(self.target_forward(next_state))
            Q_new = np.clip(Q_new, -10000, 10000)
            target[np.argmax(action)] = Q_new
        else:
            target[np.argmax(action)] = reward
        return target
    
    def target_forward(self, input: np.ndarray) -> np.ndarray:
        # calculate values after every layer
        raw_hidden_output   = self.target_weight_1.dot(input) + self.target_bias_1
        act_hidden_output   = self.__ReLU(raw_hidden_output)
        raw_output          = self.target_weight_2.dot(act_hidden_output) + self.target_bias_2
        return raw_output
    
    def __backpropagation(self, model_raw_hidden_output: np.ndarray, model_act_hidden_output: np.ndarray, model_raw_output: np.ndarray, input: np.ndarray, expected_output: np.ndarray) -> None:
        # calculate deltas
        delta_output        = model_raw_output - expected_output
        delta_weight_2      = delta_output.dot(model_act_hidden_output.T)
        delta_bias_2        = np.sum(delta_output, axis=1, keepdims=True)           # in this case: batch_size = 1, db2 = delta_output, same to bd1 below
        delta_hidden        = self.main_weight_2.T.dot(delta_output) * self.__derivative_ReLU(model_raw_hidden_output)
        delta_weight_1      = delta_hidden.dot(input.T)
        delta_bias_1        = np.sum(delta_hidden, axis=1, keepdims=True)

        # update weight and bias
        self.main_weight_1  = self.main_weight_1 - self.learning_rate * delta_weight_1
        self.main_bias_1    = self.main_bias_1 - self.learning_rate * delta_bias_1
        self.main_weight_2  = self.main_weight_2 - self.learning_rate * delta_weight_2
        self.main_bias_2    = self.main_bias_2 - self.learning_rate * delta_bias_2

    def __ReLU(self, A: np.ndarray) -> np.ndarray:
        return np.maximum(0, A)
    
    def __derivative_ReLU(self, weight: np.ndarray) -> np.ndarray:
        return weight > 0
    
    def train(self, input: np.ndarray, expected_output: np.ndarray):
        # train a single data / train short memory
        raw_hidden_output, act_hidden_output, raw_output = self.forward(input)
        self.__backpropagation(raw_hidden_output, act_hidden_output, raw_output, input, expected_output)
    
    def train_batch(self, states: np.ndarray, targets: np.ndarray) -> None:
        """
        states:  shape (batch_size,  input_size)
        targets: shape (batch_size,  output_size)
        Cập nhật weights/biases bằng backprop trên cả batch.
        """
        m = states.shape[0]
        X = states.T   # (input_size,  batch_size)
        Y = targets.T  # (output_size, batch_size)

        # --- Forward ---
        Z1 = self.main_weight_1 @ X + self.main_bias_1  # (hidden, batch)
        A1 = self.__ReLU(Z1)                            # (hidden, batch)
        Z2 = self.main_weight_2 @ A1 + self.main_bias_2 # (output, batch)

        # --- Backward (MSE loss: L = 1/2m * sum((Z2 - Y)^2)) ---
        dZ2 = (Z2 - Y)                       # (output, batch)
        dW2 = (1 / m) * dZ2 @ A1.T           # (output, hidden)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # (output, 1)

        dA1 = self.main_weight_2.T @ dZ2                    # (hidden, batch)
        dZ1 = dA1 * self.__derivative_ReLU(Z1)              # (hidden, batch)
        dW1 = (1 / m) * dZ1 @ X.T                           # (hidden, input)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)  # (hidden, 1)

        # --- Gradient descent update ---
        self.main_weight_2 -= self.learning_rate * dW2
        self.main_bias_2 -= self.learning_rate * db2
        self.main_weight_1 -= self.learning_rate * dW1
        self.main_bias_1 -= self.learning_rate * db1
        
    def set_model_path(self, model_path: str) -> None:
        self.model_path = model_path
        
    def soft_update(self, tau=0.005):
        self.target_weight_1= tau * self.main_weight_1 + (1 - tau) * self.target_weight_1
        self.target_bias_1  = tau * self.main_bias_1 + (1 - tau) * self.target_bias_1
        self.target_weight_2= tau * self.main_weight_2 + (1 - tau) * self.target_weight_2
        self.target_bias_2  = tau * self.main_bias_2 + (1 - tau) * self.target_bias_2

    def hard_update(self):
        self.target_weight_1= self.main_weight_1.copy()
        self.target_bias_1  = self.main_bias_1.copy()
        self.target_weight_2= self.main_weight_2.copy()
        self.target_bias_2  = self.main_bias_2.copy()
        
    def save(self, epoch: int, is_highscore: bool = False) -> None:
        base, ext = os.path.splitext(self.model_path)
        if is_highscore:
            save_path = f"{base}_highscore_epoch{epoch}{ext}"
        else:
            save_path = f"{base}_epoch{epoch}{ext}"
        np.savez(save_path,
                 main_weight_1=self.main_weight_1,
                 main_bias_1=self.main_bias_1,
                 main_weight_2=self.main_weight_2,
                 main_bias_2=self.main_bias_2,
                 target_weight_1=self.target_weight_1,
                 target_bias_1=self.target_bias_1,
                 target_weight_2=self.target_weight_2,
                 target_bias_2=self.target_bias_2)
    
    def load(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file '{self.model_path}' not found. Please train the model or check the path.")
        data = np.load(self.model_path)
        
        # Gán lại trọng số và bias
        self.main_weight_1 = data["main_weight_1"]
        self.main_bias_1 = data["main_bias_1"]
        self.main_weight_2 = data["main_weight_2"]
        self.main_bias_2 = data["main_bias_2"]

        self.target_weight_1= data["target_weight_1"]
        self.target_bias_1  = data["target_bias_1"]
        self.target_weight_2= data["target_weight_2"]
        self.target_bias_2  = data["target_bias_2"]