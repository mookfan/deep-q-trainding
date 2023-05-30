from collections import deque
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary





class YourModel(nn.Module):
    def __init__(self, state_size, feature_size, action_size):
        super(YourModel, self).__init__()
        self.state_size = state_size
        self.feature_size = feature_size
        self.action_size = action_size
        
        self.fc1 = nn.Linear(self.state_size * self.feature_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)
        self.fc4 = nn.Linear(3, self.action_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class Agent(object):
    def __init__(self, state_size:int, feature_size:int, batch_size:int, gamma:float=0.95, epsilon:float=0.01):
        self.gamma = gamma
        self.epsilon = epsilon
    
        self.feature_size = feature_size
        self.state_size = state_size
        self.batch_size = batch_size
        self.action_size = 3
        self._model = self.cnn_model()

        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.lr = 0.001

        # define the loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self._model.parameters(), lr=self.lr)
        

        self.myport = []
        self.memory = deque(maxlen=1000)

    
        
    def cnn_model(self):
        # model =  nn.Sequential(
        #             nn.Linear(self.state_size * self.feature_size, 16),   # input size: 14 x 2 = 28, output size: 16
        #             nn.Linear(16, 8),
        #             nn.Linear(8, 3),
        #             nn.Linear(3, self.action_size),
        #             nn.ReLU())
        # model =  nn.Sequential(
        #             nn.Linear(self.state_size * self.feature_size, 16),   # input size: 14 x 2 = 28, output size: 16
        #             nn.ReLU(),
        #             nn.Linear(16, 8),    # input size: 16, output size: 8
        #             nn.ReLU(),
        #             nn.Linear(8, 3),     # input size: 8, output size: 3
        #             nn.ReLU(),
        #             nn.Linear(3, 3),     # input size: 3, output size: 3
        #             nn.ReLU(),
        #             nn.Linear(3, 3),     # input size: 3, output size: 3
        #             nn.ReLU(),
        #             nn.Linear(3, self.action_size),     # input size: 3, output size: 3
        #         )
        # print("\n========= Model Summary ==========")
        # print("Input size: ", self.state_size * self.feature_size)
        # print(summary(model, input_size=(1, self.state_size * self.feature_size), batch_size=self.batch_size))
        # print("\n")
        # return model
        model = YourModel(self.state_size, self.feature_size, self.action_size)
        print("\n========= Model Summary ==========")
        print("Input size: ", self.state_size * self.feature_size)
        print(summary(model, input_size=(1, self.state_size * self.feature_size), batch_size=self.batch_size))
        print("\n")
        return model
        # return model
        
    def act(self, state):
        '''
            explore & exploit 
            state (self.state_size x self.feature_size)
        '''
        n = np.random.random()
        if n < self.epsilon:
            next_state = np.random.choice(np.arange(self.action_size)) # randomly select an action
        else:
            with torch.no_grad():
                y_pred = self._model(state)
                next_state = torch.argmax(y_pred)
        return int(next_state)
    
    def exp_replay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        # push data from self.memory[l-batch_size+1: l] >> mini_batch
        for i in range (l-batch_size+1, l):
            mini_batch.append(self.memory[i])
        
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * int(torch.argmax(self._model(next_state)))

            target_f = self._model(state)
            target_f[0][action] = target # update new target
            
            self.optimizer.zero_grad()

            y_pred = self._model(state)
            
            target_action = torch.argmax(target_f).unsqueeze(0) # action num
            # target_action = torch.unsqueeze(target_action, dim=1)
        
            # print(y_pred)
            # print(target_action)
            # target_action = target_action.type_as(y_pred)
            loss = self.criterion(y_pred, target_f) 
            loss.backward()
            self.optimizer.step()
            
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
