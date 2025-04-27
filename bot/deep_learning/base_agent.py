import numpy as np
import random
from collections import deque
from game.game_core import Game

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
GAMMA = 0.9
EPSILON = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.05
STOP_ON_REWARD = 1000

class BaseAgent:
    """
    Base agent class for reinforcement learning agents.
    This class is designed to be inherited by specific agent implementations.
    """

    def __init__(self, game: Game):
        self.number_of_games = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.epsilon = EPSILON
        self.mode = None
        self.model = None  # Placeholder for model, to be defined in subclasses
        self.trainer = None  # Placeholder for trainer, to be defined in subclasses
        self.game = game
        self.stop_on_reward = STOP_ON_REWARD

    def set_mode(self, mode: str):
        """
        Set the mode of the agent (training or performing).
        """
        self.mode = mode
    
    def get_state(self):
        """
        Get the current state of the game.
        This method should be implemented in subclasses to return the appropriate state.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def get_action(self, state: np.ndarray):
        """
        Get the action based on the current state.
        This method should be implemented in subclasses to return the appropriate action.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def perform_action(self, action: np.ndarray, render: bool = True):
        """
        Perform the given action in the game.
        """
        self.game.take_action(action, render)
    
    def get_reward(self) -> tuple[float, bool]:
        """
        Get the reward from the game.
        Returns a tuple of (reward, game_over).
        """
        return self.game.get_reward()

    def get_score(self) -> int:
        """
        Get the current score of the game.
        """
        return self.game.score
    
    def restart_game(self):
        self.game.restart_game()
        
    def draw_game(self):
        self.game.draw()

    def train_long_memory(self):
        if len(self.memory) <= BATCH_SIZE:
            # if have not saved over 1000 states yet
            mini_sample = self.memory
        else:
            # else pick random 1000 states to re-train
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        for current_state, action, reward, next_state in mini_sample:
            self.train_short_memory(current_state, action, reward, next_state)
    
    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)
    
    def remember(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, game_over: bool):
        self.memory.append((state, action, reward, next_state, game_over))  # popleft if MAX_MEMORY is reached