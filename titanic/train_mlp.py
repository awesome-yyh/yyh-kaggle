'''
Author: yangyahe yangyahe@midu.com
Date: 2024-07-21 02:51:50
LastEditors: yangyahe yangyahe@midu.com
LastEditTime: 2024-07-21 14:14:28
FilePath: /app/yangyahe/kaggel/titanic/sklearn.py
Description: 
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_processing import data_processing


# 加载数据
train_df = pd.read_csv("titanic/train.csv")
test_df = pd.read_csv("titanic/test.csv")
print(train_df.head())

X = data_processing(train_df)
y = train_df['Survived']

print(X.head())

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
print("==== mlp ====")
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# 定义神经网络
class TitanicNN(nn.Module):
    def __init__(self):
        super(TitanicNN, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 初始化模型、损失函数和优化器
mlp_model = TitanicNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# 训练模型
epochs = 100
for epoch in range(epochs):
    mlp_model.train()
    optimizer.zero_grad()
    output = mlp_model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# 评估模型
mlp_model.eval()
with torch.no_grad():
    output = mlp_model(X_test)
    predictions = (output > 0.5).float()
    accuracy = (predictions.eq(y_test).sum() / y_test.shape[0]).item()
    print(f'Accuracy: {accuracy:.6f}')

# Submission
test_X = data_processing(test_df)
print(test_X)

test_X = scaler.fit_transform(test_X)
test_X = torch.tensor(test_X, dtype=torch.float32)

X_test_predict = mlp_model(test_X)

X_test_predict = X_test_predict.detach().numpy().reshape(1, -1)[0]
X_test_predict = np.where(X_test_predict > 0.5, 1, 0)  
print(X_test_predict)
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': X_test_predict})
output.to_csv('titanic/submission.csv', index=False)

print("mlp Submission file has been created.")
