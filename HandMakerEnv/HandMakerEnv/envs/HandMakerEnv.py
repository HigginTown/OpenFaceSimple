import treys
import gym
from gym import spaces
from itertools import compress
import numpy as np


class HandMaker(gym.Env):
    def __init__(self):
        self.evaluator = treys.evaluator.Evaluator()
        self.deck = treys.deck.Deck()
        self.card_ints = self.deck.draw(10).sort()
        self.card_strings = [treys.Card.int_to_str(c) for c in self.card_ints]
        self.done = False
        self.reward_range = (0, 1)
        self.action_space = gym.spaces.Discrete(2)  # select from 13 cards
        self.observation_space = gym.spaces.multi_binary.MultiBinary(320)  # 32 bits * 10 cards

    def _get_obs(self):
        # the observation is the encoded representation of the cards
        # get 10 cards and convert them to binary strings and then to bits
        return np.array(
            [item for sublist in [[int(i) for i in y] for y in [f'{a:032b}' for a in self.card_ints]] for item in
             sublist], dtype='int')

    def _get_reward(self, action):
        """Return 1 minus the rank class percentage for the five card hand
        The worst hand has a value of 1, while a Royal Flush has a value of 0
        If we return 1 - this value, we return a reward in [0,1] where the 
        higher number is a better reward.
        """
        zero_is_better = self.evaluator._five(self.card_ints[:5]) < self.evaluator._five(self.card_ints[5:])
        return 1 if action == zero_is_better else 0

    def step(self, action):
        # set the state of the game to done
        self.done = True
        self.reward = self._get_reward(action)
        return self._get_obs(), self.reward, self.done, {}

    def reset(self):
        self.deck = treys.deck.Deck()
        self.done = False
        self.card_ints = self.deck.draw(10).sort()
        self.card_strings = [treys.Card.int_to_str(c) for c in self.card_ints]
        return self._get_obs()

    def render(self, mode='human'):
        pass
