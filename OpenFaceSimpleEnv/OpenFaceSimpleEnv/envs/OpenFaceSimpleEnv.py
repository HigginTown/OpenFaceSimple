from itertools import chain
import gym
import numpy as np
import treys
from gym.spaces import MultiBinary


def convert_card_to_bitlist(card):
    # requires f strings
    return [int(b) for b in "{:032b}".format(card)]

def convert_bitlist_to_int(bitlist):
    output = 0
    for bit in bitlist:
        output = output * 2 + bit
    return output


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
            f_card_bits = np.array(list(chain(*card_bits)), dtype='int')
            f_extra_bits = blank_front_cards * 32
            front_bits = np.pad(f_card_bits, (0, f_extra_bits), 'constant')
        else:
            front_bits = np.zeros(160, dtype='int')

        # and now we do the back bits
        if back_cards_num > 0:
            if back_cards_num == 1:
                b_draw = [self.deck.draw(back_cards_num)]
            else:
                b_draw = self.deck.draw(back_cards_num)
            card_bits = [convert_card_to_bitlist(card) for card in b_draw]
            b_card_bits = np.array(list(chain(*card_bits)), dtype='int')
            b_extra_bits = blank_back_cards * 32
            back_bits = np.pad(b_card_bits, (0, b_extra_bits), 'constant')
        else:
            back_bits = np.zeros(160, dtype='int')

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
        obs = np.concatenate(
            [np.zeros(320, dtype='int'), np.array(convert_card_to_bitlist(self.deck.draw(1)), dtype='int')])
        obs = np.concatenate([obs, [0, 0, 0, 0]])  # add 4 bits of 0 for the game state
        self.obs = obs
        self.done = False
        return obs

    def _get_reward(self, observation):
        """Checks if all cards have been placed and returns a positive reward if the front evaluation is 'higher'
        than the back evaluation, and returns a negative reward otherwise. If the game is not over, return 0"""
        if self.done:
            # now get the card ints from the binary data for evaluation
            front = [convert_bitlist_to_int(card) for card in [observation[32 * j:32 * (j + 1)] for j in range(0, 5)]]
            back = [convert_bitlist_to_int(card) for card in [observation[32 * j:32 * (j + 1)] for j in range(5, 10)]]
            # if 0 not in front and 0 not in back
            front_eval = self.evaluator._five(front)
            back_eval = self.evaluator._five(back)
            return 2 if front_eval > back_eval else -1

        # return 0 as the agent is playing
        else:
            return 0

    def step(self, action):
        """The action space is Discrete(2) meaning only the values of `0` and `1` are valid.
        This action will set the bits representing the player card in the correct row and position on the board

        If the board row is already full,
        then attempting to place a card there will end the episode
        the environment will return a negative reward

        :int action: binary value for action
        """

        if action == 1:
            # action == 1 points to the back row, which is the five cards, each 32 bits, after the first 160 bits
            back_row = self.obs[160:320]
            # then we chunk the array into 32 bit segments which represent a card
            card_chunks = np.split(back_row, 5)
            # we're interested in finding the first occurrence of an empty spot for the player's card
            # the below returns an array with the index positions of all the board positions
            empty_card_indices = np.where(np.array([sum(i) for i in card_chunks], dtype='int') == 0)[0]

            if empty_card_indices.size > 0:  # if this array has any elements,
                idx = empty_card_indices[0]
                # now we know the index position of the first empty board position for the card so we can set a card
                id_1 = 32 * (idx + 5)
                id_2 = 32 * (idx + 6)
                # player_card = self.obs[320:352]
                self.obs[id_1: id_2] = self.obs[320:352]

            # else (the action was 1 but we cannot find an empty board position), the game is over
            # we return a negative reward and the agent will need to reset the game
            else:
                self.done = True
                return self.obs, -10, self.done, {}

        # this is the case for the front row, similar to the above
        else:
            # the first 5 cards are stored as the first 160 bits
            front_row = self.obs[0:160]
            card_chunks = np.split(front_row, 5)
            empty_card_indices = np.where(np.array([sum(i) for i in card_chunks], dtype='int') == 0)[0]
            if empty_card_indices.size > 0:
                # now we'll find the index positions in the observation array and set the player card there
                idx = empty_card_indices[0]
                id_1 = 32 * idx
                id_2 = 32 * (idx + 1)
                # player_card = self.obs[320:352]
                self.obs[id_1: id_2] = self.obs[320:352]
            else:
                # end the game and return a negative reward if we try to play a card in a full row
                self.done = True
                return self.obs, -10, self.done, {}

        # if we have made it to this point, it means we have placed a valid card for the step
        # then we can increment the step and get a new card
        # first we get the step
        step = convert_bitlist_to_int(self.obs[-4:])
        step += 1
        step_bin = [int(b) for b in "{:04b}".format(step)]
        self.obs[-4:] = step_bin

        if step == 10:
            self.done = True

        # now set a new player card
        player_card = self.deck.draw(1)
        card_bits = convert_card_to_bitlist(player_card)
        self.obs[320:352] = card_bits

        # get the reward and return
        return self.obs, self._get_reward(self.obs), self.done, {}

    def render(self, mode='ansi'):
        print('-----------')
        print("Step: {}".format(convert_bitlist_to_int(self.obs[-4:])))
        print("Player card")
        print(treys.Card.int_to_pretty_str(convert_bitlist_to_int(self.obs[320:352])))
        print("Board")
        front = [convert_bitlist_to_int(card) for card in [self.obs[32 * j:32 * (j + 1)] for j in range(0, 5)]]
        back = [convert_bitlist_to_int(card) for card in [self.obs[32 * j:32 * (j + 1)] for j in range(5, 10)]]
        print(front)
        print(back)
        print(*[treys.Card.int_to_pretty_str(i) if i != 0 else "__" for i in front])
        print(*[treys.Card.int_to_pretty_str(i) if i != 0 else "__" for i in back])
        print('-----------')
