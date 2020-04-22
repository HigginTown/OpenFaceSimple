from itertools import chain

import gym
import numpy as np
import treys
from gym.spaces import MultiBinary


def convert_card_to_bitlist(card):
    # requires f strings
    return [int(b) for b in "{:032b}".format(card)]


def convert_bitlist_to_int(bitlist):
    return int(''.join([str(int(i)) for i in bitlist]), 2)


class OFCSObservationSpace(gym.spaces.MultiBinary):
    """Mulitbinary observation space for the OFC Game
    This class implements sample specific to OFC
    """

    def __init__(self, n):
        self.n = n
        self.shape = (n,)
        self.deck = treys.Deck()
        super(MultiBinary, self).__init__((self.n,), np.int8)

    def sample(self):
        """We need to return 356 bit vector
        320 bits for the board
        32 bits for the player card
        4 bits for the game stage
        """
        # generate a random board with a random number of cards
        # first we need to get the number of cards for each row
        # enforce that we never sample a full board
        flip = np.random.choice(2)
        if flip:
            front_cards_num = np.random.choice(5)  # pick up to 4 cards
            back_cards_num = np.random.choice(6)  # pick up to 5 cards
        else:
            front_cards_num = np.random.choice(6)  # pick up to 5 cards
            back_cards_num = np.random.choice(5)  # pick up to 4 cards

        blank_front_cards = 5 - front_cards_num
        blank_back_cards = 5 - back_cards_num

        # first we need to get the front card bits
        # notice we need to add a variable number of zeroes depending on how many blanks we have
        if front_cards_num > 0:
            if front_cards_num == 1:
                f_draw = [self.deck.draw(front_cards_num)]
            else:
                f_draw = self.deck.draw(front_cards_num)
            card_bits = [convert_card_to_bitlist(card) for card in f_draw]
            f_card_bits = np.array(list(chain(*card_bits)))
            f_extra_bits = blank_front_cards * 32
            front_bits = np.pad(f_card_bits, (0, f_extra_bits), 'constant')
        else:
            front_bits = np.zeros(160)

        # and now we do the back bits
        if back_cards_num > 0:
            if back_cards_num == 1:
                b_draw = [self.deck.draw(back_cards_num)]
            else:
                b_draw = self.deck.draw(back_cards_num)
            card_bits = [convert_card_to_bitlist(card) for card in b_draw]
            b_card_bits = np.array(list(chain(*card_bits)))
            b_extra_bits = blank_back_cards * 32
            back_bits = np.pad(b_card_bits, (0, b_extra_bits), 'constant')
        else:
            back_bits = np.zeros(160)

        # now we can do the bits for the observation card
        player_card = convert_card_to_bitlist(self.deck.draw(1))

        # and finally we can get the game state
        # notice that the game state cannot exceed 8
        # a game state of 0 means there are 0 cards played, whereas 9 means all but one cards are set
        # a game state of 10 ends the episode
        game_state = convert_card_to_bitlist(back_cards_num + front_cards_num)[-4:]
        # print([front_bits, back_bits, game_state])
        # now we can concatenate all of these

        return np.concatenate([front_bits, back_bits, player_card, game_state])


class OpenFaceSimpleEnv(gym.Env):
    """Simple Open Face Chinese inspired environment

    INTRODUCTION
    ------------
    The goal of the game is to place each of 10 cards successively into one of two rows.
    After the ten cards are placed, each 'row' of five cards is passed to the evaluator.
    Then the rows are compared.
      If row 0 evaluates to a stronger hand than row 1, a reward of -10 is returned.
      If row 1 evaluates to a stronger hand than row 0, a base reward of 1 is awarded, plus bonus points for stronger hands.

    Observation Space and Action Space
    -----------------------------------

    The observation space is the union of the board and the card to be placed.
    The board starts with 10 empty slots where the agent will place cards over 10 actions.
    Since each card can be represented by a 32bit vector, the observation space can be modeled as
        10 spots + 1 card to be placed = 10 * 32 + 1 * 32 = 352 bit list

    Then we need to model the stage of the game. Since there are 10 phases, we can use a 4 bit vector.
        This is an extra 4 bits

    For gym, this is MultiBinary(356)

    The action space is Discrete(2). For each step, the agent must decide to play in row 0 or row 1.
    """

    def __init__(self):
        self.deck = treys.Deck()
        self.evaluator = treys.Evaluator()
        self.reward_range = (-1, 1)  # we will process the reward to fit in [-1,1] from [-10,10]
        self.metadata = {'render_modes': ['ansi']}
        self.observation_space = OFCSObservationSpace(356)
        self.action_space = gym.spaces.Discrete(2)
        self.done = False
        self.obs = self.reset()

    def reset(self):
        """Returns a new observation and resets the env"""
        self.deck = treys.Deck()
        # create a new starting observation
        obs = np.concatenate([np.zeros(320), np.array(convert_card_to_bitlist(self.deck.draw(1)))])
        obs = np.concatenate([obs, [0, 0, 0, 0]])  # add 4 bits of 0 for the game state
        self.obs = obs
        self.done = False
        return obs

    def _get_reward(self, observation):
        # check if the game is over and if so return a reward
        if convert_bitlist_to_int(observation[-4:]) == 10:
            self.done = True
            # now get the card ints from the binary data for evaluation
            front = [convert_bitlist_to_int(card) for card in [observation[32 * j:32 * (j + 1)] for j in range(0, 5)]]
            back = [convert_bitlist_to_int(card) for card in [observation[32 * j:32 * (j + 1)] for j in range(5, 10)]]
            if 0 not in front and 0 not in back:
                front_eval = self.evaluator._five(front)
                back_eval = self.evaluator._five(back)
                return 100 if front_eval > back_eval else -100
        return 0

    def step(self, action):
        """Each action is a binary choice.
        This action will take the player card and place it on the board"""
        # if action is True, this means we place in the back row
        # if the row is full, ignore the action
        step = convert_bitlist_to_int(self.obs[-4:])
        if step >= 10:
            self.done = True
            return self.obs, 0, self.done, {}
        if action == 1:
            back_row = self.obs[160:320]
            card_chunks = np.split(back_row, 5)
            first_empty_card_index = np.where(np.array([sum(i) for i in card_chunks]) == 0)[0]
            if first_empty_card_index.shape[0] > 0:
                idx = int(first_empty_card_index[0])
                # now we know the index position of the card
                # we can set the card there
                player_card = list(self.obs[320:352])
                tmp = list(self.obs)
                tmp[32 * (idx + 5): 32 * (idx + 6)] = player_card
                self.obs = np.array(tmp)
                # now we need to increment the step
                step += 1
                if step == 10:
                    self.done = True
                step_bin = [int(b) for b in "{:04b}".format(step)]
                self.obs[-4:] = step_bin

        elif action == 0:
            front_row = self.obs[0:160]
            card_chunks = np.split(front_row, 5)
            first_empty_card_index = np.where(np.array([sum(i) for i in card_chunks]) == 0)[0]
            if first_empty_card_index.shape[0] > 0:
                idx = int(first_empty_card_index[0])
                # now set the card
                player_card = list(self.obs[320:352])
                tmp = list(self.obs)
                tmp[32 * idx:32 * (idx + 1)] = player_card
                self.obs = np.array(tmp)
                # now we need to increment the step
                step += 1
                if step == 10:
                    self.done = True
                step_bin = [int(b) for b in "{:04b}".format(step)]
                self.obs[-4:] = step_bin

        # notice that if the action is invalid (ie we can't find a space for the card), then we ignore the action
        # effectively we just get a new card

        # now we need to set a new player card
        player_card = self.deck.draw(1)
        card_bits = convert_card_to_bitlist(player_card)
        self.obs[320:352] = card_bits

        # now we can return
        # print(self.done, step)
        return self.obs, self._get_reward(self.obs), self.done, {}

    def render(self, mode='ansi'):
        print('-----------')
        print("Step: {}".format(convert_bitlist_to_int(self.obs[-4:])))
        print("Board")
        front = [convert_bitlist_to_int(card) for card in [self.obs[32 * j:32 * (j + 1)] for j in range(0, 5)]]
        back = [convert_bitlist_to_int(card) for card in [self.obs[32 * j:32 * (j + 1)] for j in range(5, 10)]]
        print(*[treys.Card.int_to_pretty_str(i) if i != 0 else "__" for i in front])
        print(*[treys.Card.int_to_pretty_str(i) if i != 0 else "__" for i in back])
        print('-----------')
