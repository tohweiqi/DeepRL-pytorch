import gym
import numpy as np
import pickle
from typing import Tuple

class Serialize_Env(gym.ObservationWrapper):
    '''
    Simple wrapper to add the save and load functionality
    '''
    def __init__(self, env, training=True):
        super(Serialize_Env, self).__init__(env)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation):
        return observation

    def save(self, fname):
        return

    # @classmethod
    def load(self, filename):
        return