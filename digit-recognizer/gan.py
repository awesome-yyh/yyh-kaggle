import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.view(-1, 28 * 28))

def save_generated_images(generator, epoch, device, num_images=25):
    z = torch.randn(num_images, 100).to(device)
    fake_images = generator(z)
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    fake_images = fake_images.cpu().data

    fig, axes = plt.subplots(5, 5, figsize=(5, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(fake_images[i].squeeze(), cmap='gray')
        ax.axis('off')

    if not os.path.exists('generated_images'):
        os.makedirs('generated_images')

    plt.savefig(f'generated_images/epoch_{epoch+1}.png')
    plt.close()


def train_gan(generator, discriminator, dataloader, num_epochs=50, lr=0.0002):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss()
    optimizerG = optim.Adam(generator.parameters(), lr=lr)
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            z = torch.randn(batch_size, 100).to(device)
            fake_imgs = generator(z)
            real_loss = criterion(discriminator(real_imgs), real_labels)
            fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
            d_loss = real_loss + fake_loss

            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            # Train Generator
            z = torch.randn(batch_size, 100).to(device)
            fake_imgs = generator(z)
            g_loss = criterion(discriminator(fake_imgs), real_labels)

            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()

        print(f'Epoch [{epoch+1}/{num_epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}')

        # 每个epoch结束后生成一些图像并保存
        save_generated_images(generator, epoch, device)

# 设置数据集和数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 初始化生成器和判别器
generator = Generator()
discriminator = Discriminator()

# 训练GAN
train_gan(generator, discriminator, train_dataloader, num_epochs=50)
