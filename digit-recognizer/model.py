import torch
import torch.nn as nn


device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')


# 定义 MLP 网络
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义 CNN 网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        # (28 + 2*2 - 5 + 1) // 1 = 28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)  # (14 + 2*2 -5 + 1) // 2 = 14
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 28/2 = 14
        x = self.pool(torch.relu(self.conv2(x)))  # 14/2 = 7
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CnnPpo(nn.Module):
    def __init__(self):
        super(CnnPpo, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        # (28 + 2*2 - 5 + 1) // 1 = 28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)  # (14 + 2*2 -5 + 1) // 2 = 14
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(128, 1)
        self.device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, x):
        x = x.to(self.device)
        # print(x, x.shape)
        x = self.pool(torch.relu(self.conv1(x)))  # 28/2 = 14
        x = self.pool(torch.relu(self.conv2(x)))  # 14/2 = 7
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        action_logits = self.fc2(x)
        state_value = self.fc3(x)
        return action_logits, state_value

# 定义 RNN 网络
class RNN(nn.Module):
    def __init__(self, input_size=28, hidden_size=512, num_layers=2, num_classes=10):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 初始隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播 RNN
        out, _ = self.rnn(x, h0)
        
        # 解码最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


# 定义 双向RNN 网络
class BiRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(BiRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向 RNN

    def forward(self, x):
        x = x.view(x.size(0), 28, 28)  # [batch_size, 28, 28]
        h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_dim).to(device)  # 2 for bidirectional
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


# 定义双向 LSTM 模型
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向 LSTM

    def forward(self, x):
        x = x.view(x.size(0), 28, 28)  # [batch_size, 28, 28]
        h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_dim).to(device)  # 2 for bidirectional
        c0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


# 定义双向 GRU 模型
class BiGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(BiGRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向 GRU

    def forward(self, x):
        x = x.view(x.size(0), 28, 28)  # [batch_size, 28, 28]
        h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_dim).to(device)  # 2 for bidirectional
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(AttentionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平图像 [batch_size, 28*28]
        x = torch.relu(self.fc1(x))  # [batch_size, hidden_dim]
        attention_weights = torch.softmax(self.attention(x), dim=1)  # [batch_size, 1]
        attention_applied = x * attention_weights  # [batch_size, hidden_dim]
        output = self.fc2(attention_applied)  # [batch_size, num_classes]
        return output


class SelfAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        self.hidden_dim = hidden_dim
        super(SelfAttentionModel, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平图像 [batch_size, 28*28]
        
        # 计算 Q, K, V
        Q = self.query(x)  # [batch_size, hidden_dim]
        K = self.key(x)    # [batch_size, hidden_dim]
        V = self.value(x)  # [batch_size, hidden_dim]
        
        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # [batch_size, batch_size]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, batch_size]
        
        # 应用注意力权重于 V
        attention_applied = torch.matmul(attention_weights, V)  # [batch_size, hidden_dim]
        
        # 最后的全连接层
        output = self.fc(attention_applied)  # [batch_size, num_classes]
        return output


class TransformerDecoderModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes):
        super(TransformerDecoderModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 28, model_dim))  # 位置编码
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 28, 28)  # [batch_size, 28, 28]
        x = self.embedding(x) + self.pos_encoder  # [batch_size, 28, model_dim]
        x = x.permute(1, 0, 2)  # [28, batch_size, model_dim] (Transformer expects [seq_len, batch_size, model_dim])

        # 使用一个全零的 tensor 作为 memory
        memory = torch.zeros(28, x.size(1), x.size(2)).to(x.device)

        x = self.decoder(x, memory)
        x = x.permute(1, 0, 2)  # [batch_size, 28, model_dim]
        x = x.mean(dim=1)  # 在序列长度维度上取平均值 [batch_size, model_dim]
        x = self.fc(x)  # [batch_size, num_classes]
        return x


# 定义 Transformer 网络
class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.position_encoding = nn.Parameter(torch.zeros(1, 28*28, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # 将输入展平为二维张量
        x = x.view(x.size(0), -1, 28*28)  # [batch_size, input_dim, 28*28]
        x = self.embedding(x) + self.position_encoding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # 平均池化
        x = self.fc(x)
        return x
