import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torchvision import datasets, transforms


class MNISTEnv(gym.Env):
    def __init__(self):
        super(MNISTEnv, self).__init__()
        # Load the MNIST dataset
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        
        self.action_space = spaces.Discrete(10)  # 10 classes for MNIST digits 0-9
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 28, 28), dtype=np.float32)
        
        self.current_index = 0

    def reset(self):
        self.current_index = np.random.randint(0, len(self.mnist))
        image, label = self.mnist[self.current_index]
        self.current_label = label
        return image.numpy()

    def step(self, action):
        """_summary_

        Args:
            action (int): 当前状态下采取的动作, 即图片的预测标签

        Returns:
            reward: 采取该动作的奖励
            next_state: 采取动作后到达的下一个状态（state: 当前状态）
            done: 当前回合是否结束
        """
        # print(f"current_index: {self.current_index}")
        reward = 1 if action == self.current_label else -1
        
        if reward == 1:
            done = True
        else:
            done = False
        
        info = {}
        
        # Move to the next image
        # next_image = self.mnist[self.current_index][0]

        return self.reset(), reward, done, info
    
    def render(self, mode='human'):
        pass

    def close(self):
        pass
