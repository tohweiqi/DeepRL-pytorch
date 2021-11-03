import gym
import numpy as np
import pickle
from typing import Tuple
from gym.spaces import Box

class RLBench_Wrapper(gym.ObservationWrapper):
    '''
    Observation Wrapper for the RLBench environment to only output 1 of the 
    camera views during training/testing instead of a dictionary of all camera views
    Observation space is in the shape (128, 128, 3), while actual observations are tweaked to be
    of the shape (3, 128, 128) for ease of conversion into tensor
    '''
    def __init__(self, env, view):
        '''
        Args:
            view (str): Dictionary key to specify which camera view to use. 
                        RLBench observation comes in a dictionary of
                        ['state', 'left_shoulder-rgb', 'right_shoulder-rgb', 'wrist-rgb', 'front-rgb'
                        'left_shoulder-rgbd', 'right_shoulder-rgbd', 'wrist-rgbd', 'front-rgbd']
        '''
        super(RLBench_Wrapper, self).__init__(env)
        self.view, self.viewtype = view.split('-')
        if self.viewtype == 'rgb':
            # swap (128, 128, 3) into (3, 128, 128) for torch input
            H, W, C = self.observation_space[self.view+'_rgb'].shape
            self.observation_space = Box(0.0, 1.0, (C, H, W), dtype=np.float32)
        elif self.viewtype == 'rgbd':
            # swap (128, 128, 3) and (128, 128) into (4, 128, 128) for torch input
            H, W, C = self.observation_space[self.view+'_rgb'].shape
            self.observation_space = Box(0.0, 1.0, (C+1, H, W), dtype=np.float32)
        else:
            self.observation_space = self.observation_space[view]

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation):
        #print(observation.keys())
        if self.viewtype == 'rgb':
            return observation[self.view+'_rgb'].transpose([2, 0, 1])
        elif self.viewtype == 'rgbd':
            rgbd_obs = np.dstack((observation[self.view+'_rgb'], observation[self.view+'_depth']))
            print(rgbd_obs.transpose([2, 0, 1]).shape)
            return rgbd_obs.transpose([2, 0, 1])
        elif self.viewtype == 'depth':
            return observation[self.view+'_depth']
    
    def save(self, fname):
        return

    # @classmethod
    def load(self, filename):
        return
