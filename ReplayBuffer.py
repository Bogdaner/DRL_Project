import numpy as np
import torch

class ReplayBuffer():
    def  __init__(self, stateDim, actionDim, maxSize=int(1e6)):
        self.maxSize = maxSize
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((maxSize, stateDim))
        self.action = np.zeros((maxSize, actionDim))
        self.nextState = np.zeros((maxSize, stateDim))
        self.reward = np.zeros((maxSize, 1))
        self.notDone = np.zeros((maxSize, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, nextState, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.nextState[self.ptr] = nextState
        self.reward[self.ptr] = reward
        self.notDone[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.maxSize
        self.size = min(self.size + 1, self.maxSize)

    def sample(self, batchSize):
        ind = np.random.randint(0, self.size, size=batchSize)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.nextState[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.notDone[ind]).to(self.device)
        )
