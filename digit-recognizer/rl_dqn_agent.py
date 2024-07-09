import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from model import CNN


class DQNAgent:
    def __init__(self):
        self.device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
        self.model = CNN().to(self.device)
        self.target_model = CNN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.memory = deque(maxlen=10000)
        
        self.gamma = 0.99  # 折扣因子，用于权衡当前奖励和未来奖励的关系。值越接近 1，表示越重视未来的奖励。
        self.epsilon = 1.0  # 探索率，表示智能体采取随机动作（探索）的概率。初始值为 1.0，表示智能体刚开始时完全随机探索。
        self.epsilon_decay = 0.995  # 探索率衰减系数，每次更新后 epsilon 乘以这个系数，逐渐减少探索率。
        self.epsilon_min = 0.01  # 探索率的最小值，即 epsilon 不会低于这个值，确保智能体在训练后期仍有一定概率进行探索。

    def remember(self, state, action, reward, next_state, done):
        """将agent收集的经验存储在经验池中
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """输入图片，返回label

        Args:
            state (np.array): 图片

        Returns:
            int: label
        """
        if np.random.rand() <= self.epsilon:
            # print("开始随机探索")
            return random.randrange(10)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state.to(self.device))
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        """经验回放：从记忆库中随机抽取一批经验，并使用这些经验来训练 Q 网络。
        经验回放通过打乱训练数据，提高了训练的稳定性和效率。

        Args:
            batch_size (int): 回放的经验数
        """
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            
            target = self.model(state.to(self.device)).clone().detach()
            if done:
                # 回合结束，目标 Q 值等于即时奖励
                target[0][action] = reward
            else:
                # 目标 Q 值等于即时奖励加上折扣后的下一个状态的最大 Q 值。
                t = self.target_model(next_state.to(self.device)).max(1)[0].item()
                target[0][action] = reward + self.gamma * t
            
            # 更新 Q 网络：使用目标 Q 值和当前 Q 网络输出的 Q 值计算损失, 并更新网络
            self.optimizer.zero_grad()
            output = self.model(state.to(self.device))
            loss = nn.MSELoss()(output, target)
            loss.backward()
            self.optimizer.step()
        
        # 更新探索率: 逐渐减少探索率，确保智能体在训练初期进行更多探索，而在训练后期更多地利用学习到的策略。
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())