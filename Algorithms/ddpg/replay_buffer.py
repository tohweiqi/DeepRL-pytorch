import numpy as np
from collections import deque
import random
import torch
import pickle

class ReplayBuffer:
    '''
    A FIFO experience replay buffer to store 
    '''
    def __init__(self, size):
        """
        Args:
            size (integer): The size of the replay buffer.
        """
        size = int(size)
        self.buffer = deque(maxlen=size)
        self.max_size = size
        self.size = 0

    def append(self, state, action, reward, next_state, terminal):
        '''
        Args:
            state (Numpy ndarray): The state.              
            action (integer): The action.s
            reward (float): The reward.
            next_state (Numpy ndarray): The next state. 
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
        '''
        if self.size < self.max_size:
            self.size += 1
        self.buffer.append([state, action, reward, next_state, terminal])

    def sample(self, batch_size):
        '''
        Randomly sample experiences from replay buffer
        Args:
            batch_size (int): number of samples to retrieve from replay buffer
        Returns:
            A list of transition tuples including state, action, reward, next state and terminal
        '''
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        return [np.asarray(x, dtype="float32") for x in list(zip(*batch))]
        
    def size(self):
        '''
        Return the current size of the replay buffer
        '''
        return len(self.buffer)

    def save(self, filename):
        '''
        Save the replay buffer as a python object using pickle
        Args:
            filename (str): full path to the saved file to save the replay buffer to
        '''
        with open(filename, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, filename):
        '''
        Load the replay buffer as a python object using pickle
        Args:
            filename (str): full path to the saved file to load the replay buffer from
        '''
        with open(filename, 'rb') as f:
            self.buffer = pickle.load(f)
        assert self.buffer.maxlen == self.max_size, "Attempted to load buffer with different max size"
