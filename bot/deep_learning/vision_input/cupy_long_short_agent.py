import cupy as cp
import random
import itertools
from datetime import datetime

if __name__ == "__main__":
    # only re-direct below if running this file
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from bot.deep_learning.models.cupy_model import Model
from game.game_core import Game
from bot.deep_learning.base_agent import BaseAgent
from configs.bot_config import DATE_FORMAT
from utils.bot_helper import plot_training_progress

MAX_MEMORY = 10000
MIN_MEMORY = 1000 # at least 1000 samples in memory before training
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.99
EPSILON = 1
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01
NETWORK_UPDATE_FREQ = 1000

IMG_SIZE = 80 # 80 x 80 pixel^2

USE_SOFT_UPDATE = False
TAU = 0.005

MODEL_PATH = 'saved_files/vision/cupy/long_short.npz'
GRAPH_PATH = 'saved_files/vision/cupy/long_short.png'
LOG_PATH = 'saved_files/vision/cupy/long_short.log'

class VisionCupyLongShortAgent(BaseAgent):

    def __init__(self, game: Game, load_saved_model: bool = False):
        super().__init__(game)
        self.epsilon = EPSILON
        self.model = Model(2*IMG_SIZE**2, 512, 9, LEARNING_RATE, DISCOUNT_FACTOR, MODEL_PATH, load_saved_model)
        #warning: the number of neurals in first layer must match the size of game.get_state()
        
        self.network_update_freq = NETWORK_UPDATE_FREQ # Update target network every 250 steps

    def get_state(self) -> cp.ndarray:
        """
        Get the current game state and reshape it to 28x1 for model input
        example: array([1, 1, 0, 0, 0, 1, 0, ...0])
        """
        np_state = self.game.get_state(is_heuristic=False, is_vision=True, method="cupy")
        return cp.asarray(np_state)

    def get_action(self, state: cp.ndarray) -> cp.ndarray:
        action = cp.zeros((9, ), dtype=cp.float64)
        if self.mode == "train":
            # decise to take a random action or not
            if random.random() < self.epsilon:
                # if yes pick a random action
                action[random.randint(0, 8)] = 1
            else:
                # if not model will predict the action
                action[cp.argmax(self.model.predict(state))] = 1
        elif self.mode == "perform":
            # always use model to predict action in pridict action / always predict
            action[cp.argmax(self.model.predict(state))] = 1
        return action

    def train_short_memory(self, current_state: cp.ndarray, action: cp.ndarray, reward: float, next_state: cp.ndarray, game_over: bool):
        target = self.model.compute_target(current_state, action, reward, next_state, game_over)
        self.model.train(current_state, target)

    def train_long_memory(self):
        if len(self.memory) <= MIN_MEMORY:
            # if have not saved over 1000 states yet
            mini_sample = self.memory
        else:
            # else pick random 1000 states to re-train
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        for current_state, action, reward, next_state, game_over in mini_sample:
            self.train_short_memory(current_state, action, reward, next_state, game_over)
        if USE_SOFT_UPDATE:
            self.model.soft_update()
    
    def train(self, render: bool = False, show_graph: bool = True):
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
                self.perform_action(int(cp.argmax(action)), render)

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
                self.model.save(episode)
            
            self.train_long_memory()
            
            if not USE_SOFT_UPDATE and step_count >= self.network_update_freq:
                # update target network
                self.model.hard_update()
                step_count = 0
            
            # Update graph every 5 games
            if self.number_of_games % 5 == 0:
                cpu_scores = cp.asnumpy(scores_per_episode)
                plot_training_progress(cpu_scores, title='Vision LongShort Training', show_graph=show_graph, save_dir=GRAPH_PATH)
                
            # reduce epsilon / percentage of random move
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(self.epsilon, MIN_EPSILON)

    def perform(self, render: bool = True):
        self.set_mode("perform")
        
        self.load_model()

        while True:
            # get the current game state
            state = self.get_state()

            # get the model predict move
            action = self.get_action(state)

            # perform selected move
            self.perform_action(int(cp.argmax(action)), render)

            # check if game over or not
            _, game_over = self.get_reward()

            # restart game if game over
            if game_over:
                self.restart_game()

            # use pygame to control FPS and UPS
            self.game.clock.tick(60)
    
    def load_model(self):
        self.model.load()

if __name__ == '__main__':
    agent = VisionCupyLongShortAgent(Game())

    mode = "train"

    if mode == "train":
        agent.train()
    elif mode == "perform":
        agent.perform()