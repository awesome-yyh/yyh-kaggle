import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import tqdm
from rl_env import MNISTEnv
from model import CnnPpo


# Define the PPO agent
class PPOAgent:
    def __init__(self, env, policy_net, lr=1e-4, gamma=0.99, eps_clip=0.2, K_epochs=10):
        self.device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
        
        self.env = env
        self.policy_net = policy_net.to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

    def get_action(self, state):
        """输入模型，得到动作
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        action_logits, state_value = self.policy_net(state)
        action_prob = torch.softmax(action_logits, dim=-1)
        action = torch.multinomial(action_prob, num_samples=1).item()
        logprob = torch.log(action_prob.squeeze(0)[action])
        return action, logprob, state_value

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + (1 - dones[step]) * self.gamma * R
            returns.insert(0, R)
        return returns
    
    def update(self, memory):
        # print("开始更新")
        states = torch.FloatTensor(np.array(memory['states'])).to(self.device)
        actions = torch.LongTensor(np.array(memory['actions'])).to(self.device)
        rewards = np.array(memory['rewards'])
        dones = np.array(memory['dones'])
        old_logprobs = torch.FloatTensor(np.array(memory['logprobs'])).to(self.device)
        values = torch.FloatTensor(np.array(memory['values'])).to(self.device)
        
        next_value = self.policy_net(torch.FloatTensor(memory['next_state']).unsqueeze(0))[1].item()
        returns = self.compute_returns(rewards, dones, values, next_value)
        returns = torch.FloatTensor(returns).detach().to(self.device)

        loss_all, loss_policy_all, loss_value_all = 0, 0, 0
        for idx in range(self.K_epochs):
            # print(f"idx: {idx}")
            # print(states.shape)
            action_logits, state_values = self.policy_net(states)
            action_probs = torch.softmax(action_logits, dim=-1)
            dist_entropy = torch.distributions.Categorical(action_probs).entropy().mean()
            # print(actions.unsqueeze(1).device)
            new_logprobs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())

            ratios = torch.exp(new_logprobs - old_logprobs)
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()  # 策略损失, 保证新的参数和旧的参数的差距不会太大
            critic_loss = 0.5 * nn.MSELoss()(state_values, returns) # 衡量评论家预期收益和真实收益之间的差距
            loss = policy_loss + critic_loss - 0.01 * dist_entropy # _ + _ + 熵正则化
            
            loss_all += loss
            loss_policy_all += policy_loss
            loss_value_all += critic_loss
            # print(f"loss: {loss}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_all / self.K_epochs, loss_policy_all / self.K_epochs, loss_value_all / self.K_epochs

    def train(self, total_timesteps):
        memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'logprobs': [],
            'values': [],
            'next_state': None
        }
        state = self.env.reset()
        for t in tqdm.tqdm(range(total_timesteps)):
            action, logprob, state_value = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            memory['states'].append(state)
            memory['actions'].append(action)
            memory['rewards'].append(reward)
            memory['dones'].append(done)
            memory['logprobs'].append(logprob.item())
            memory['values'].append(state_value.item())

            state = next_state
            if done:
                state = self.env.reset()

            if (t + 1) % 2048 == 0:  # 更新间隔
                memory['next_state'] = state
                loss_mean, loss_policy_mean, loss_value_mean = self.update(memory)
                print(f"loss_mean: {loss_mean}, loss_policy_mean: {loss_policy_mean}, loss_value_mean: {loss_value_mean}")
                memory = {
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'dones': [],
                    'logprobs': [],
                    'values': [],
                    'next_state': None
                }

# 创建环境和模型
env = MNISTEnv()
model = CnnPpo()

# 创建PPO Agent并训练
ppo_agent = PPOAgent(env, model)
ppo_agent.train(total_timesteps=100*10000)

# 测试模型
state = env.reset()
acc = 0
print(state.shape)
for _ in range(100):
    action, _, _ = ppo_agent.get_action(state)
    next_state, reward, done, _ = env.step(action)
    # print(f"Action: {action}, action: {action}, Reward: {reward}")
    state = next_state if not done else env.reset()
    if done:
        acc += 1
print(f"test acc: {acc/100}")

df = pd.read_csv("./test.csv", skiprows=0, sep=',', encoding='utf_8_sig', header=0, index_col=None)
data = df.values.reshape(-1, 28, 28).astype('float32')
data_tensor = torch.tensor(data)
# print(data_tensor.shape)

# device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
batch_size = 128
ppo_agent.policy_net.eval()
preds = []
with torch.inference_mode():
    for i in range(0, len(data_tensor), 1):
        batch_images_tensor = data_tensor[i].unsqueeze(0)
        # print(batch_images_tensor.shape)
        action, _, _ = ppo_agent.get_action(batch_images_tensor)
        preds.append(action)

image_ids = [idx + 1 for idx in range(len(data_tensor))]
result_df = pd.DataFrame({
    "ImageID": image_ids,
    "Label": preds
})
result_df.to_csv("./submission.csv", index=False)

print("./submission.csv")