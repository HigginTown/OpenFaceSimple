import treys
import gym
from itertools import compress
import numpy as np


class HandClassificationEnv(gym.Env):
    def __init__(self):
        self.evaluator = treys.evaluator.Evaluator()
        self.deck = treys.deck.Deck()
        self.card_ints = self.deck.draw(5)
        self.rank_class = self.evaluator.get_rank_class(self.evaluator._five(self.card_ints))
        self.card_strings = [treys.Card.int_to_str(c) for c in self.card_ints]
        self.done = False
        self.reward_range = (-1, 1)
        self.action_space = gym.spaces.discrete.Discrete(9)  # select one of 9 hands from 5
        self.observation_space = gym.spaces.multi_binary.MultiBinary(160)  # 32 bits * 5 cards

    def _get_obs(self):
        # the observation is the encoded representation of the cards
        # get 13 cards and convert them to binary strings
        return np.array(
            [item for sublist in [[int(i) for i in y] for y in [f'{a:032b}' for a in self.card_ints]] for item in
             sublist])

    def _get_reward(self, action):
        """If the choice matches the rank class, get one point. Otherwise, minus 1. 
        """
        return (9 - self.rank_class) / 9 if action == self.rank_class - 1 else -1

    def step(self, action):
        # set the state of the game to done
        self.done = True
        self.reward = self._get_reward(action)

        return self._get_obs(), self.reward, self.done, {}

    def reset(self):
        self.deck = treys.deck.Deck()
        self.done = False
        self.card_ints = self.deck.draw(5)
        self.rank_class = self.evaluator.get_rank_class(self.evaluator._five(self.card_ints))
        return self._get_obs()
