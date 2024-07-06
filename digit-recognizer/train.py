import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from dataset import MnistDataSet
from model import *


device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

batch_size = 128
dataset_full = MnistDataSet("./digit-recognizer/train.csv")
train_dataset, eval_dataset = random_split(dataset_full, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
print(len(train_dataset), len(eval_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, pin_memory=True, prefetch_factor=10, persistent_workers=True, shuffle=False, drop_last=False)

eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=1, pin_memory=True, prefetch_factor=10, persistent_workers=True, shuffle=False, drop_last=False)

# net = MLP()  # 96.76%, 53.5818w
net = CNN()  # 98.38%, 42.1642w
# net = RNN(input_size=28, hidden_size=256, num_layers=4, num_classes=10)  #  92.27%, 47.0538w
# net = BiRNNModel(input_dim=28, hidden_dim=256, num_layers=2, num_classes=10)  # 92.62%, 54.5802w
# net = BiLSTMModel(input_dim=28, hidden_dim=128, num_layers=2, num_classes=10)  # 96.61, 55.9626w
# net = BiGRUModel(input_dim=28, hidden_dim=128, num_layers=2, num_classes=10)  # 96.280%, 42.0362w

# net = SelfAttentionModel(input_dim=28*28, hidden_dim=256, num_classes=10)
# net = AttentionModel(input_dim=28*28, hidden_dim=1024, num_classes=10).to(device)  # 97.21%, 81.5115 w
# net = TransformerModel(input_dim=784, embed_dim=128, num_heads=4, num_layers=2, num_classes=10)  # 95.75% 138.817w
# net = TransformerDecoderModel(input_dim=28, model_dim=128, num_heads=4, num_layers=2, num_classes=10)  # 96.05% 198.657w
# net = MoE(input_size=784, output_size=10, num_experts=10, hidden_size=128, noisy_gating=True, k=4)  # 94.35, 103.338w

print(net)
print(f"总参数量: {sum([p.nelement() for p in net.parameters()]) / 1e4} w")
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-5)
# linear_scheduler = get_linear_schedule_with_warmup(
    # optimizer, num_warmup_steps=50, num_training_steps=2000)

# 训练网络
num_epochs = 10
print_interval = 100
net.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        batch = {key: value.to(device) for key, value in batch.items()}
        
        optimizer.zero_grad()
        
        # outputs, aux_loss = net(inputs.view(inputs.shape[0], -1))  # MoE
        # outputs = net(inputs.squeeze(1))  # RNN
        outputs = net(batch["feature"])
    
        loss = criterion(outputs, batch["label"])
        # total_loss = loss + aux_loss  # MOE
        total_loss = loss
        total_loss.backward()
        optimizer.step()
        
        # linear_scheduler.step()

        running_loss += loss.item()
        if (i+1) % print_interval == 0:  # 每 100 个 mini-batch 打印一次
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    if len(eval_dataloader) > 0:
        correct = 0
        total = 0
        net.eval()
        with torch.inference_mode():
            for batch in eval_dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                
                # outputs, _ = net(inputs.view(inputs.shape[0], -1)) # MOE需要
                # outputs = net(inputs.squeeze(1))  # RNN
                outputs = net(batch['feature'])
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch['label'].size(0)
                correct += (predicted == batch['label']).sum().item()

        print(f'epoch: {epoch}, Accuracy of the network on the 10000 test images: {100 * correct / total:.6f} %')

print('Finished Training')


df = pd.read_csv("./digit-recognizer/test.csv", skiprows=0, sep=',', encoding='utf_8_sig', header=0, index_col=None)
data = df.values.reshape(-1, 1, 28, 28).astype('float32')
data_tensor = torch.tensor(data)

batch_size = 128
net.eval()
preds = []
with torch.inference_mode():
    for i in range(0, len(data_tensor), batch_size):
        batch_images_tensor = data_tensor[i:i + batch_size].to(device)
        logits = net(batch_images_tensor)
        cur_preds = torch.argmax(logits, dim=1)
        preds.extend(cur_preds.cpu().numpy())

image_ids = [idx + 1 for idx in range(len(data_tensor))]
result_df = pd.DataFrame({
    "ImageID": image_ids,
    "Label": preds
})
result_df.to_csv("./submission.csv", index=False)

print("./digit-recognizer/submission.csv")
