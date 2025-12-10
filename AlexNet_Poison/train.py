from model import AlexNet, select_subset, fetch_datasets, plot_misclassified_images, plot_correctly_classified_images
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
# clean_rate 和 poison_rate 分别表示干净样本和投毒样本的比例
clean_rate =1
poison_rate =0
# 从库中获取训练
trainset_all = datasets.MNIST(root='./data', download=True, train=True)
trainset = select_subset(trainset_all)
all_datasets = fetch_datasets(full_dataset=trainset_all, trainset=trainset, ratio=[poison_rate, clean_rate])
poison_trainset = all_datasets['poisonTrain']
clean_trainset = all_datasets['cleanTrain']
all_trainset = poison_trainset.__add__(clean_trainset)

# 获取测试集
clean_test_all = datasets.MNIST(root='./data', download=True, train=False)
clean_test = select_subset(clean_test_all)
clean_testset = []
for img, label in clean_test:
    img = np.array(img)  # 转换为一个Numpy数组
    img = torch.from_numpy(img / 255.0)  # 白一化，将像素值从0-255缩放到0-1之间，将这个归一化后的NumPy数组转为PyTorch张量
    clean_testset.append((img, label))

# 数据加载器
trainset_dataloader = DataLoader(dataset=all_trainset, batch_size=64, shuffle=True)

print('--------------------开始对模型投毒--------------------')

# 实验以 AlexNet
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
net = AlexNet().to(device)

# 定义交叉熵函数
loss_fn = torch.nn.CrossEntropyLoss().to(device)

# 记录干净的准确率
clean_acc_list = []

# 使用带有动量的 Adam 优化器对模型优化
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
epoch = 5
clean_correct = 0   # 记录投毒后模型准确率

file = open("training_log.txt", "a", encoding="utf-8")


for epoch in range(epoch):
    running_loss = 0.0
    for index, (imgs, labels) in enumerate(trainset_dataloader, 0):
        # 获取输入数据
        imgs = imgs.unsqueeze(1)
        imgs = imgs.type(torch.FloatTensor)
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()      # 将梯度置为0
        outputs = net(imgs)        # 前向传播
        loss = loss_fn(outputs, labels)  # 损失
        loss.backward()            # 反向传播

        optimizer.step()           # 更新参数
        running_loss += loss.item()

        # 输出每一轮loss值
    print('Epoch: {},loss:{}'.format(epoch + 1, running_loss))
    file.write("Epoch:" + str(epoch + 1) + ",loss:" + str(running_loss)+'\n')
    file.flush()

# 在循环体外部测试每轮准确率
print("测试每一轮干净样本准确率:Epoch" + str(epoch + 1) + "------------------------------")
clean_correct = 0
for img, label in clean_testset:
    img = img.type(torch.FloatTensor)
    img = img.unsqueeze(0).unsqueeze(0).to(device)
    pred = net(img)
    pred = torch.reshape(pred, shape=(10,))
    top_pred = torch.argmax(pred)
    if top_pred.item() == label:
        clean_correct += 1

clean_acc = clean_correct / len(clean_testset) * 100
clean_acc_list.append(clean_acc)

print("干净样本准确率为:" + str(clean_acc) + "%\n")

# 展示错误分类的图片
plot_misclassified_images(net, clean_testset, device)

# 展示正确分类的图片
plot_correctly_classified_images(net, clean_testset, device, num_images=10)

# 关闭文件
file.close()

#绘制结果图
plt.rcParams['font.size'] = 16
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(clean_acc_list) + 1), clean_acc_list,
         label='Accuracy', marker='o', linestyle='-')
plt.title(f'poison_rate={poison_rate}')
plt.xlabel('训练轮数(epoch)')
plt.ylabel('准确率(%)')
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.show()
