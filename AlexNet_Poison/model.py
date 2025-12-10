import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Subset

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 由于 MNIST 为 28x28 ，而最初 AlexNet 的输入图片是 227x227 的。所以网络层数和参数需要调节
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu3 = nn.ReLU()
        self.fc6 = nn.Linear(256 * 3 * 3, 1024)
        self.fc7 = nn.Linear(1024, 512)
        self.fc8 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        return x

def select_subset(dataset, ratio=1 / 10):
    subset_size = int(len(dataset) * ratio)
    indices = np.random.choice(range(len(dataset)), subset_size, replace=False)
    return Subset(dataset, indices)


# 展示正确分类的图片
def plot_correctly_classified_images(model, dataset, device, num_images=10):
    model.eval()
    correctly_classified_imgs = []

    for img, label in dataset:
        img = img.type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img)
        pred_label = torch.argmax(pred).item()
        if pred_label == label:
            correctly_classified_imgs.append((img.cpu().squeeze(), label, pred_label))
            if len(correctly_classified_imgs) >= num_images:
                break

    plt.figure(figsize=(10, 10))
    for i, (img, true_label, pred_label) in enumerate(correctly_classified_imgs):
        plt.subplot(5, 2, i + 1)
        plt.imshow(img.numpy(), cmap='gray')
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_misclassified_images(model, dataset, device, num_images=10):
    model.eval()
    misclassified_imgs = []
    for img, label in dataset:
        img = img.type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img)
        pred_label = torch.argmax(pred).item()

        if pred_label != label:
            misclassified_imgs.append((img.cpu().squeeze(), label, pred_label))
            if len(misclassified_imgs) >= num_images:
                break

    plt.figure(figsize=(10, 10))
    for i, (img, true_label, pred_label) in enumerate(misclassified_imgs):
        plt.subplot(5, 2, i + 1)
        plt.imshow(img.numpy(), cmap='gray')
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 对训练数据集进行切分，ratio用于控制干净样本与投毒样本比例
def fetch_datasets(full_dataset, trainset, ratio):  # 1个用法
    character = [[] for i in range(len(full_dataset.classes))]
    for index in trainset.indices:
        img, label = full_dataset[index]
        character[label].append(img)

    poison_trainset = []
    clean_trainset = []
    target = 0

    for i, data in enumerate(character):
        num_poison_train_inputs = int(len(data) * ratio[0])

        for img in data[:num_poison_train_inputs]:
            # 对投毒样本添加标签
            target = random.randint(a=0, b=9)
            poison_img = img
            poison_img = torch.from_numpy(np.array(poison_img) / 255.0)
            poison_trainset.append((poison_img, target))

        for img in data[num_poison_train_inputs:]:
            # 干净数据标签不变
            img = np.array(img)
            img = torch.from_numpy(img / 255.0)
            clean_trainset.append((img, i))

    result_datasets = {}
    result_datasets['poisonTrain'] = poison_trainset
    result_datasets['cleanTrain'] = clean_trainset

    return result_datasets
