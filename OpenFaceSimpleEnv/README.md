# Open Face Chinese Poker Gym environments
This repo will build a new text based gym environment based on Open Face Chinese Poker

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
