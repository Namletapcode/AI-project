if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# from bot.heuristic_dodge import HeuristicDodgeBot
from bot.supervised_learning.model import Model
from game.game_core import Game
from collections import deque
import numpy as np
import random

MAX_MEMORY = 100_000
MAX_SAMPLE_SIZE = 5_000
LEARNING_RATE = 0.001

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

class Supervised_Agent:
    def __init__(self):
        self.game = Game()
        self.model = Model(28, 256, 9, LEARNING_RATE)
        # self.coach = HeuristicDodgeBot(self.game)
        self.coach = Coach()
        self.memory = deque(maxlen=MAX_MEMORY)
        self.number_of_games = 0

    def get_state(self) -> np.ndarray:
        state = self.game.get_state(is_heuristic=False)
        return state.reshape(len(state), 1)
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros((9,), dtype=np.float64)
        model_result = self.model.forward(state)[2]
        action[np.argmax(model_result)] = 1
        return action
    
    def perform_action(self, action: np.ndarray, render: bool = True):
        self.game.take_action(action, render)

    def get_score(self) -> int:
        return self.game.score
    
    def get_coach_action(self, state: np.ndarray) -> np.ndarray:
        """state = self.game.get_state(True)
        coach_action = self.coach.get_action(state)
        return coach_action"""
        return self.coach.get_action(state)
    
    def is_game_over(self) -> bool:
        return self.game.get_reward()[1]
    
    def train_short_memory(self, state: np.ndarray, expected_action: np.ndarray):
        self.model.train(state, expected_action)

    def train_long_memory(self):
        if len(self.memory) < MAX_SAMPLE_SIZE:
            mini_sample = self.memory
        else:
            mini_sample = random.sample(self.memory, MAX_SAMPLE_SIZE)
        for state, expected_action in mini_sample:
            self.model.train(state, expected_action)

    def remember(self, state: np.ndarray, expected_action: np.ndarray):
        self.memory.append((state, expected_action))

    def save(self):
        self.model.save()

    def reset_game(self):
        self.game.restart_game()
    
    def train(self, render: bool = False):

        while True:

            state = self.get_state()

            coach_action = self.get_coach_action(state)

            self.perform_action(coach_action, render)

            game_over = self.is_game_over()

            if not game_over:
                coach_action = coach_action.reshape(9, 1)
                self.train_short_memory(state, coach_action)
                self.remember(state, coach_action)

            else:
                self.number_of_games += 1
                print("Game:", self.number_of_games,"Score:", self.get_score())
                self.train_long_memory()
                if self.number_of_games % 10 == 0:
                    self.save()
                self.reset_game()

    def perform(self, render:bool = True):
        
        while True:

            state = self.get_state()

            agent_action = self.get_action(state)

            self.perform_action(agent_action, render)

            game_over = self.is_game_over()

            if game_over:
                self.number_of_games += 1
                print("Game:", self.number_of_games,"Score:", self.get_score())
                self.reset_game()

            self.game.clock.tick(60)

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
