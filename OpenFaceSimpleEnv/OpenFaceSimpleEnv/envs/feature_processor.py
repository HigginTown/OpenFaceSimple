import numpy as np


def observation_processor(observation):
    """Augments the observation space to encode logical XOR

    :param observation: np.array        this is the observation from the environment, should be a 356 binary vector

    :returns augmented_obs: np.array    the augmented observation with 64 additional bits, 420 bit vector
    """
    raise NotImplemented
