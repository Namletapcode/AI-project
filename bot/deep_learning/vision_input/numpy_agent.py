import numpy as np
import random
import itertools
from datetime import datetime

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    
from bot.deep_learning.models.numpy_model import Model
from bot.deep_learning.base_agent import BaseAgent
from game.game_core import Game
from configs.bot_config import DodgeAlgorithm, DATE_FORMAT
from utils.bot_helper import plot_training_progress, show_numpy_to_image

MAX_MEMORY = 10000
MAX_SAMPLE_SIZE = 1000
LEARNING_RATE = 0.0001
GAMMA = 0.9
EPSILON = 1
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.01
NETWORK_UPDATE_FREQ = 1000

IMG_SIZE = 50 # 50 x 50 pixel^2

USE_SOFT_UPDATE = False
TAU = 0.005

MODEL_PATH = 'saved_files/vision_numpy/vision_numpy_model.npz'
GRAPH_PATH = 'saved_files/vision_numpy/vision_numpy_training.png'
LOG_PATH = 'saved_files/vision_numpy/vision_numpy_log.log'

class VisionNumpyAgent(BaseAgent):
    def __init__(self, game: Game, load_saved_model: bool = False):
        super().__init__(game)
        self.epsilon = EPSILON
        self.model = Model((IMG_SIZE ** 2) * 2, 9, 9, LEARNING_RATE, MODEL_PATH, load_saved_model) #warning: the number of neurals in first layer must match the size of game.get_state()
        
        self.network_update_freq = NETWORK_UPDATE_FREQ # Update target network every 250 steps
        
        self.reset_self_img()

    def reset_self_img(self):
        self.img_01 = np.zeros((IMG_SIZE ** 2, 1), dtype=np.float64)
        self.img_02 = np.zeros((IMG_SIZE ** 2, 1), dtype=np.float64)

    def get_state(self) -> np.ndarray: # get game state. stack of two consecutive screenshot around player
        return self.game.get_state(is_heuristic=False, is_vision=True, is_numpy=True)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        move = np.zeros((9, ), dtype=np.float64)
        if self.mode == "train":
            # decise to take a random move or not
            if random.random() < self.epsilon:
                # if yes pick a random move
                move[random.randint(0, 8)] = 1
            else:
                # if not model will predict the move
                move[np.argmax(self.model.forward(state)[2])] = 1
        elif self.mode == "perform":
            # always use model to predict move in pridict move / always predict
            move[np.argmax(self.model.forward(state)[2])] = 1
        return move
    
    def restart_game(self):
        self.game.restart_game()
        self.reset_self_img()

    def train_short_memory(self, current_state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, game_over: bool):
        target = self.convert(current_state, action, reward, next_state, game_over)
        self.model.train(current_state, target)

    def train_long_memory(self):
        if len(self.memory) <= MAX_SAMPLE_SIZE:
            # if have not saved over 1000 states yet
            mini_sample = self.memory
        else:
            # else pick random 1000 states to re-train
            mini_sample = random.sample(self.memory, MAX_SAMPLE_SIZE)
        for current_state, action, reward, next_state, game_over in mini_sample:
            self.train_short_memory(current_state, action, reward, next_state, game_over)

    def convert(self, current_state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, game_over: bool) -> np.ndarray:
        # use simplified Bellman equation to calculate expected output
        if not game_over:
            target = self.model.forward(current_state)[2]
            Q_new = reward + GAMMA * np.max(self.model.target_forward(next_state))
            Q_new = np.clip(Q_new, -10000, 10000)
            target[np.argmax(action)] = Q_new
        else:
            target = self.model.forward(current_state)[2]
            target[np.argmax(action)] = reward
        return target
    
    def train(self, render: bool = True, show_graph: bool = True):
        self.set_mode("train")
        rewards_per_episode = []
        scores_per_episode = []
        best_score = -999999
        step_count = 0
            
        for episode in itertools.count():
            self.restart_game()
            self.number_of_games += 1
            # get the current game state
            current_state = self.get_state()

            game_over = False
            episode_reward = 0
            episode_score = 0
            
            while not game_over and episode_score < self.stop_on_score:
                
                # get the move based on the state
                action = self.get_action(current_state)

                # perform action in game
                self.perform_action(action, render)

                # get the new state after performed action
                next_state = self.get_state()

                # get the reward of the action
                reward, game_over = self.get_reward()
                
                episode_reward += reward
                episode_score = self.get_score()

                # train short memory with the action performed
                self.train_short_memory(current_state, action, reward, next_state, game_over)

                # remember the action and the reward
                self.remember(current_state, action, reward, next_state, game_over)
                
                step_count += 1
                current_state = next_state

            # if game over then train long memory and start again
            rewards_per_episode.append(episode_reward)
            scores_per_episode.append(episode_score)
            
            if episode_score >= self.stop_on_score:
                print(f"Game {self.number_of_games} finished after {episode} episodes")
                break
            if episode_score > best_score:
                log_message = f"{datetime.now().strftime(DATE_FORMAT)} Episode {episode}: New best score: {episode_score:0.1f} ({(episode_score-best_score)/best_score*100:+.1f}%)"
                print(log_message)
                with open(LOG_PATH, 'a') as log_file:
                    log_file.write(log_message + '\n')
                best_score = episode_score
                self.model.save()
            
            self.train_long_memory()
            
            if not USE_SOFT_UPDATE and step_count >= self.network_update_freq:
                # update target network
                self.model.hard_update()
                step_count = 0
            
            # Update graph every 5 games
            if self.number_of_games % 5 == 0:
                plot_training_progress(scores_per_episode, title='Vision_numpy Training', show_graph=show_graph, save_dir=GRAPH_PATH)
                
            # reduce epsilon / percentage of random move
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(self.epsilon, MIN_EPSILON)

    def perform(self, render: bool = True):
        self.set_mode("perform")

        while True:
            # get the current game state
            state = self.get_state()

            # get the model predict move
            action = self.get_action(state)

            # perform selected move
            self.perform_action(action, render)

            # check if game over or not
            _, game_over = self.get_reward()

            # restart game if game over
            if game_over:
                self.restart_game()

            # use pygame to control FPS and UPS
            self.game.clock.tick(60)
            
    def load_model(self):
        self.model.load()
        
    def visualize_state(self):
        show_numpy_to_image(self.img_02, IMG_SIZE)

if __name__ == '__main__':
    agent = VisionNumpyAgent(Game())
    mode = "train"

    if mode == "train":
        agent.train()
    elif mode == "perform":
        agent.perform()