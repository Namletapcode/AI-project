if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from collections import deque
import numpy as np
import random
import itertools
from datetime import datetime
from bot.supervised_learning.model import Model
from game.game_core import Game
from bot.deep_learning.base_agent import BaseAgent
from configs.bot_config import DATE_FORMAT
import matplotlib.pyplot as plt

MAX_MEMORY = 100_000
MIN_MEMORY = 5_000
BATCH_SIZE = 64
LEARNING_RATE = 0.001

MODEL_PATH = 'saved_files/supervised/supervised_model.npz'
GRAPH_PATH = 'saved_files/supervised/supervised_training.png'
LOG_PATH = 'saved_files/supervised/supervised_log.log'
class Coach:
    def __init__(self):
        self.wall_penalty_multiple = 1.1

    def get_action(self, state: np.ndarray) -> np.ndarray:
        if state.shape == (28, 1):
            state = state.ravel()  # faster than reshape

        action = np.zeros(9, dtype=np.float64)
        action[8] = -1000

        # Predefine danger scores
        danger_scores = [16, 8, 4, 2, 1]  # level 5, 4, 3, 2, 1

        # Handle state[0] to state[23] with neighbor spreading
        for level in range(3):  # Levels 0 (0-7), 1 (8-15), 2 (16-23)
            offset = level * 8
            current_zone = state[offset:offset+8]
            active_indices = np.flatnonzero(current_zone)

            for idx in active_indices:
                # Main danger score at current position
                action[idx] -= danger_scores[level]
                action[(idx + 4) % 8] -= danger_scores[level + 2]
                # Neighbor danger scores (next lower level)
                left = (idx - 1) % 8  # wrap around 0-7
                right = (idx + 1) % 8
                action[[left, right]] -= danger_scores[level + 1]
                left = (idx - 2) % 8  # wrap around 0-7
                right = (idx + 2) % 8
                action[[left, right]] -= danger_scores[level + 2]

        # Handle level 1 manually (last 4 state values)
        if state[24]:
            action[2] -= danger_scores[0] * self.wall_penalty_multiple
            action[[1, 3]] -= danger_scores[2]
        if state[25]:
            action[0] -= danger_scores[0] * self.wall_penalty_multiple
            action[[7, 1]] -= danger_scores[2]
        if state[26]:
            action[6] -= danger_scores[0] * self.wall_penalty_multiple
            action[[5, 7]] -= danger_scores[2]
        if state[27]:
            action[4] -= danger_scores[0] * self.wall_penalty_multiple
            action[[3, 5]] -= danger_scores[2]

        # If all first 8 elements are zero, set action[8] to 1
        if np.all(action[:8] == 0):
            action[8] = 1

        # Randomly select one of the maximum indices
        choice = np.random.choice(np.flatnonzero(action == np.max(action)))

        # Set the chosen action
        final_action = np.zeros((9,), dtype=np.float64)
        final_action[choice] = 1

        return final_action

class Supervised_Agent(BaseAgent):
    def __init__(self, game: Game, load_saved_model: bool = False):
        super().__init__(game)
        self.model = Model(28, 256, 9, LEARNING_RATE, MODEL_PATH, load_saved_model)
        self.coach = Coach()
        self.losses = []

    def get_state(self) -> np.ndarray:
        return self.game.get_state(is_heuristic=False, is_vision=False, method="numpy")
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros((9,), dtype=np.float64)
        model_result = self.model.forward(state)[2]
        action[np.argmax(model_result)] = 1
        return action

    def get_action_idx(self, state: np.ndarray) -> int:
        return np.argmax(self.model.forward(state)[2])
    
    def get_coach_action(self, state: np.ndarray) -> np.ndarray:
        """state = self.game.get_state(True)
        coach_action = self.coach.get_action(state)
        return coach_action"""
        return self.coach.get_action(state)
    
    def train_short_memory(self, state: np.ndarray, expected_action: np.ndarray):
        loss = self.model.train(state, expected_action)
        self.losses.append(loss)

    def train_long_memory(self):
        if len(self.memory) <= MIN_MEMORY:
            return
        mini_batch = random.sample(self.memory, BATCH_SIZE)
        states = np.zeros((BATCH_SIZE, 28), dtype=np.float64)
        expected_actions = np.zeros((BATCH_SIZE, 9), dtype=np.float64)
        for i, (state, expected_action) in enumerate(mini_batch):
            states[i] = state.flatten()
            expected_actions[i] = expected_action.flatten()
        # Chuyển thành numpy array và train batch
        batch_loss = self.model.train_batch(
            states,             # shape (batch_size, 28)
            expected_actions    # shape (batch_size, 9)
        )
        self.losses.append(batch_loss)

    def remember(self, state: np.ndarray, expected_action: np.ndarray):
        self.memory.append((state, expected_action))
    
    def train(self, render: bool = False, show_graph: bool = True):
        self.set_mode("train")
        lowest_loss = float('inf')  # Track best (lowest) loss instead of highest score
        step_count = 0
        episode_avg_losses = []
        
        for episode in itertools.count():
            self.restart_game()
            self.number_of_games += 1
            # get the current game state
            current_state = self.get_state()

            game_over = False
            
            while not game_over:
                
                coach_action = self.get_coach_action(current_state)

                # perform action in game
                self.perform_action(np.argmax(coach_action), render)

                # get the new state after performed action
                next_state = self.get_state()

                # get the reward of the action
                _, game_over = self.get_reward()

                coach_action = coach_action.reshape(9, 1)
                self.train_short_memory(current_state, coach_action)
                self.remember(current_state, coach_action)
                
                step_count += 1
                current_state = next_state
            
            episode_avg_loss = np.mean(self.losses[-step_count:]) if step_count > 0 else float('inf')
            episode_avg_losses.append(episode_avg_loss)
            if episode_avg_loss < lowest_loss:
                log_message = f"{datetime.now().strftime(DATE_FORMAT)} Episode {episode}: New best loss: {episode_avg_loss:.6f}"
                print(log_message)
                with open(LOG_PATH, 'a') as log_file:
                    log_file.write(log_message + '\n')
                    
            if episode % 20 == 0:
                self.model.save(episode, False)
                
            self.train_long_memory()
            
            # Update graph every 5 games
            if self.number_of_games % 2 == 0:
                self.__plot_loss(episode_avg_losses, show_graph)
    def __plot_loss(self, average_loss, show_graph: bool):
        """Plot training loss over time"""
        if len(average_loss) == 0:
            return
        
        if show_graph:
            plt.ion()   # Ensure interactive mode is on if not headless
        else:
            plt.ioff()  # Turn off interactive mode
        plt.cla()
        plt.title('Episode average Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.plot(average_loss)
        plt.grid(True)
        
        if show_graph:
            plt.pause(0.001)
        
        plt.savefig(GRAPH_PATH)

    def perform(self, render:bool = True):
        
        while True:

            state = self.get_state()

            agent_action = self.get_action(state)

            self.perform_action(np.argmax(agent_action), render)

            _, game_over = self.get_reward()

            if game_over:
                self.number_of_games += 1
                print("Game:", self.number_of_games,"Score:", self.get_score())
                self.restart_game()

            self.game.clock.tick(60)
            
    def load_model(self):
        self.model.load()

    def bench(self):
        # manually used to find the best multiple

        rate = 0.5
        rate_record = 0
        mean_score_record = 0

        while rate <= 2.0:
            number_of_games = 0
            score_sum = 0

            while number_of_games <= 100:
                state = self.get_state()
                action = self.get_coach_action(state)
                self.perform_action(action)
                game_over = self.is_game_over()

                if game_over:
                    number_of_games += 1
                    score_sum += self.get_score()
                    self.reset_game()

            mean_score = score_sum / 100
            if mean_score > mean_score_record:
                mean_score_record = mean_score
                rate_record = rate

            print("Rate:", rate, "Average score:", mean_score)
            rate += 0.1

        print("Best rate:", rate_record, "Average score:", mean_score_record)

if __name__ == "__main__":
    spv_agent = Supervised_Agent()
    spv_agent.perform()
