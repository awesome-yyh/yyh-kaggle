import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from rl_env import MNISTEnv
from rl_dqn_agent import DQNAgent
from dataset import MnistDataSet


env = MNISTEnv()
agent = DQNAgent()
episodes = 1000
batch_size = 16

device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

dataset_full = MnistDataSet("./train.csv")
train_dataset, eval_dataset = random_split(dataset_full, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=1, pin_memory=True, prefetch_factor=10, persistent_workers=True, shuffle=False, drop_last=False)


for e in range(episodes):
    state = env.reset()
    state = np.array(state)
    total_reward = 0
    
    for time in range(10):
        action = agent.act(state)
        
        next_state, reward, done, _ = env.step(action)
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        # print(f"当前索引: {env.current_index}, done: {done}, time: {time}")
        if done:
            agent.update_target_model()
            break
    
    agent.replay(batch_size)
    
    print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    if len(eval_dataloader) > 0:
        correct = 0
        total = 0
        agent.target_model.eval()
        with torch.inference_mode():
            for batch in eval_dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                
                outputs = agent.target_model(batch['feature'])
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch['label'].size(0)
                correct += (predicted == batch['label']).sum().item()

        print(f'epoch: {e+1}/{episodes}, Accuracy of the network on the {total} test images: {100 * correct / total:.6f} %')


print('Finished Training')


df = pd.read_csv("./test.csv", skiprows=0, sep=',', encoding='utf_8_sig', header=0, index_col=None)
data = df.values.reshape(-1, 1, 28, 28).astype('float32')
data_tensor = torch.tensor(data)

batch_size = 128
agent.target_model.eval()
preds = []
with torch.inference_mode():
    for i in range(0, len(data_tensor), batch_size):
        batch_images_tensor = data_tensor[i:i + batch_size].to(device)
        logits = agent.target_model(batch_images_tensor)
        cur_preds = torch.argmax(logits, dim=1)
        preds.extend(cur_preds.cpu().numpy())

image_ids = [idx + 1 for idx in range(len(data_tensor))]
result_df = pd.DataFrame({
    "ImageID": image_ids,
    "Label": preds
})
result_df.to_csv("./submission.csv", index=False)

print("./submission.csv")
