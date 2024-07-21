from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoTokenizer, BertForSequenceClassification, BertTokenizer, RobertaTokenizer, DebertaTokenizer, RobertaForSequenceClassification, DebertaForSequenceClassification, AutoModelForSequenceClassification, AutoModelForCausalLM, GPT2ForSequenceClassification, LlamaForSequenceClassification, T5ForSequenceClassification
from transformers import MambaConfig, MambaForCausalLM

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from dataset import BertDataSet
from plot_f1 import displayConfusionMatrix


device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
batch_size = 8

# base_model = "/data/app/yangyahe/base_model/google-bert-bert-base-uncased"  # 83, 77 # pure 42.9, 83, 77
# base_model = "/data/app/yangyahe/base_model/FacebookAI-roberta-base"  # 81, 75  # pure 55.7, 83, 78
base_model = "/data/app/yangyahe/base_model/microsoft-deberta-v3-base"  # 81, 76  # pure 42, 83, 77

# base_model = "/data/app/yangyahe/base_model/IDEA-CCNL-Wenzhong-GPT2-110M"  # pure 55, 79, 71
# base_model = "/data/app/yangyahe/base_model/openai-community-gpt2"  # 1.2y 83, 78  # pure 48.8, 82, 77 # left 42.9, 82, 78, 
# base_model = "/data/app/yangyahe/base_model/EleutherAI-gpt-neo-125m" # 82, 76  # pure 43 82, 77 # left 45, 82, 76
# base_model = "/data/app/yangyahe/base_model/facebook-opt-125m"  # 80, 71  # pure 82 77 # left 47, 82, 76
# base_model = "/data/app/yangyahe/base_model/Felladrin-Llama-68M-Chat-v1"  # 81, 74  # pure 48, 81, 76  # left 46, 81, 76

# base_model = "/data/app/yangyahe/base_model/google-gemma-2b-it" # oom
# base_model = "/data/app/yangyahe/base_model/bigscience-bloomz-560m"  # 78, 75  # pure 42, 79， 69
# base_model = "/data/app/yangyahe/base_model/state-spaces-mamba2-130m"
# base_model = "/data/app/yangyahe/base_model/state-spaces-mamba-130m-hf"
# base_model = "/data/app/yangyahe/base_model/google-flan-t5-small"
# base_model = "/data/app/yangyahe/base_model/google-flan-t5-base"
# base_model = "/data/app/yangyahe/base_model/google-flan-t5-large"

config = AutoConfig.from_pretrained(base_model)
if not config.pad_token_id:
    print("添加pad_token_id")
    config.pad_token_id = config.eos_token_id
# print(config)

net = AutoModelForSequenceClassification.from_pretrained(base_model, config=config)  # , ignore_mismatched_sizes=True
# net = MambaForCausalLM.from_pretrained(base_model, config=config)

# print(net)
print(f"\n{base_model}; 总参数量: {sum([p.nelement() for p in net.parameters()]) / 1e8} y")
net.to(device)

dataset_full = BertDataSet("./nlp-getting-started/train.csv", base_model=base_model)
train_dataset, eval_dataset = random_split(dataset_full, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
print("train and test: ", len(train_dataset), len(eval_dataset))


optimizer = optim.Adam(net.parameters(), lr=1e-6)  # 1e-5
# linear_scheduler = get_linear_schedule_with_warmup(
    # optimizer, num_warmup_steps=50, num_training_steps=2000)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, pin_memory=True, prefetch_factor=10, persistent_workers=True, shuffle=False, drop_last=False)

eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=1, pin_memory=True, prefetch_factor=10, persistent_workers=True, shuffle=False, drop_last=False)

y_true_eval, y_pred_eval = [], []
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

    print(f'init, Accuracy of the network on the 10000 test images: {100 * correct / total:.6f} %')

# 训练网络
num_epochs = 5
print_interval = 100
net.train()
y_true_train, y_pred_train = [], []
y_true_eval, y_pred_eval = [], []
for epoch in tqdm.tqdm(range(num_epochs)):    
    total_loss = 0.0
    correct = 0
    total = 0
    for i, batch in enumerate(train_dataloader):
        batch = {key: value.to(device) for key, value in batch.items()}
        
        optimizer.zero_grad()
        
        outputs = net(**batch)
        
        loss = outputs.loss

        y_true_train.extend(batch["labels"].cpu().numpy())
        y_pred_train.extend(torch.max(outputs.logits, 1)[1].cpu().numpy())
    
        _, predicted = torch.max(outputs.logits, 1)
        total += batch['labels'].size(0)
        correct += (predicted == batch['labels']).sum().item()
        
        loss.backward()
        optimizer.step()
        
        # linear_scheduler.step()

        total_loss += loss.item()
        if (i+1) % print_interval == 0:  # 每 100 个 mini-batch 打印一次
            print(f'train [Epoch {epoch + 1}, Batch {i + 1}] loss: {total_loss / 100:.3f}')
            total_loss = 0.0
    print(f'train epoch: {epoch + 1}, Accuracy of the network on the 10000 train images: {100 * correct / total:.6f} %')

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

        print(f'test epoch: {epoch + 1}, Accuracy of the network on the 10000 test images: {100 * correct / total:.6f} %')

print('Finished Training')
print("f1 train:")
displayConfusionMatrix(y_true_train, y_pred_train, "training", "./nlp-getting-started/training.png")
if y_true_eval:
    print("f1 test:")
    displayConfusionMatrix(y_true_eval, y_pred_eval, "Validation", "./nlp-getting-started/Validation.png")

test_df = pd.read_csv("./nlp-getting-started/test.csv", skiprows=0, sep=',', encoding='utf_8_sig', header=0, index_col=None)
tokenizer = AutoTokenizer.from_pretrained(base_model)
# tokenizer.padding_side = 'left'
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
max_length = 160+20

net.eval()
preds = []
start_time = time.time()
with torch.inference_mode():
    for i in range(0, len(test_df), batch_size):
        text_ori = test_df["text"][i:i + batch_size].tolist()
        new_text = [f"Please determine if the content described in this tweet involves a real disaster: {text}. Answer:" for text in text_ori]
        # new_text = text_ori
        
        text_encode_dict = tokenizer(new_text, add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
        batch = {key: value.to(device) for key, value in text_encode_dict.items()}
        
        outputs = net(**batch)
        
        logits = outputs.logits
        cur_preds = torch.argmax(logits, dim=1)
        preds.extend(cur_preds.cpu().numpy())

print(f"test time: {time.time()-start_time}")
result_df = pd.DataFrame({
    "id": test_df["id"].tolist(),
    "target": [i for i in preds]
})
result_df.to_csv("./nlp-getting-started/submission.csv", index=False)

print("./nlp-getting-started/submission.csv")
