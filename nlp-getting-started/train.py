from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, BertTokenizer
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from dataset import BertDataSet
from plot_f1 import displayConfusionMatrix


device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

batch_size = 128
base_model = "/data/app/yangyahe/base_model/google-bert-bert-base-uncased"
dataset_full = BertDataSet("./nlp-getting-started/train.csv", base_model=base_model)
train_dataset, eval_dataset = random_split(dataset_full, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
print("train and test: ", len(train_dataset), len(eval_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, pin_memory=True, prefetch_factor=10, persistent_workers=True, shuffle=False, drop_last=False)

eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=1, pin_memory=True, prefetch_factor=10, persistent_workers=True, shuffle=False, drop_last=False)

net = BertForSequenceClassification.from_pretrained(base_model)
print(net)
print(f"总参数量: {sum([p.nelement() for p in net.parameters()]) / 1e8} y")
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-5)
# linear_scheduler = get_linear_schedule_with_warmup(
    # optimizer, num_warmup_steps=50, num_training_steps=2000)

# 训练网络
num_epochs = 3
print_interval = 100
net.train()
y_true_train, y_pred_train = [], []
y_true_eval, y_pred_eval = [], []
for epoch in range(num_epochs):
    total_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        batch = {key: value.to(device) for key, value in batch.items()}
        
        optimizer.zero_grad()
        
        outputs = net(**batch)
        
        loss = outputs.loss

        y_true_train.extend(batch["labels"].cpu().numpy())
        y_pred_train.extend(torch.max(outputs.logits, 1)[1].cpu().numpy())
    
        loss.backward()
        optimizer.step()
        
        # linear_scheduler.step()

        total_loss += loss.item()
        if (i+1) % print_interval == 0:  # 每 100 个 mini-batch 打印一次
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {total_loss / 100:.3f}')
            total_loss = 0.0

    if len(eval_dataloader) > 0:
        correct = 0
        total = 0
        net.eval()
        with torch.inference_mode():
            for batch in eval_dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                
                outputs = net(**batch)
                
                y_true_eval.extend(batch["labels"].cpu().numpy())
                y_pred_eval.extend(torch.max(outputs.logits, 1)[1].cpu().numpy())
                
                _, predicted = torch.max(outputs.logits, 1)
                total += batch['labels'].size(0)
                correct += (predicted == batch['labels']).sum().item()

        print(f'epoch: {epoch}, Accuracy of the network on the 10000 test images: {100 * correct / total:.6f} %')

print('Finished Training')
displayConfusionMatrix(y_true_train, y_pred_train, "training", "./nlp-getting-started/training.png")
if y_true_eval:
    displayConfusionMatrix(y_true_eval, y_pred_eval, "Validation", "./nlp-getting-started/Validation.png")

test_df = pd.read_csv("./nlp-getting-started/test.csv", skiprows=0, sep=',', encoding='utf_8_sig', header=0, index_col=None)
tokenizer = BertTokenizer.from_pretrained(base_model)
max_length = 160

batch_size = 128
net.eval()
preds = []
with torch.inference_mode():
    for i in range(0, len(test_df), batch_size):
        text_encode_dict = tokenizer(test_df["text"][i:i + batch_size].tolist(), add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
        batch = {key: value.to(device) for key, value in text_encode_dict.items()}
        
        outputs = net(**batch)
        
        logits = outputs.logits
        cur_preds = torch.argmax(logits, dim=1)
        preds.extend(cur_preds.cpu().numpy())

result_df = pd.DataFrame({
    "id": test_df["id"].tolist(),
    "target": preds
})
result_df.to_csv("./nlp-getting-started/submission.csv", index=False)

print("./nlp-getting-started/submission.csv")
