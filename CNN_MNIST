import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

# Hyper Parameters
EPOCH = 3
BATCH_SIZE = 100
LR = 1e-3

# 数据集MNIST，从torchvision.datasets中下载
train_data = torchvision.datasets.MNIST(
    root = './Data/',
    train = True,
    # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
    transform = torchvision.transforms.ToTensor(),
    # 本地已有数据，若无则改成True
    download = False
)
test_data = torchvision.datasets.MNIST(
    root = './Data/',
    train = False,
    transform = torchvision.transforms.ToTensor()
)

# Torch.torch.utils.data.DataLoader数据加载器
# 如果dataset是装满数据的水箱，DataLoader就是一个起到调节数据出来方式的水龙头（如批处理，乱序）
train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)
test_loader = Data.DataLoader(dataset = test_data, batch_size = BATCH_SIZE, shuffle = False)

# 构建卷积神经网络
class CNN(nn.Module):
    # in_dim : 输入数据的维度 out_class : 输出数据的分类数
    def __init__(self, in_dim , out_class):
        # super函数将CNN的对象转换为父类Module的对象
        super(CNN, self).__init__()
        # Conv:卷积层 ReLU:激励函数 MaxPool:池化层 目的：提取特征
        self.conv = nn.Sequential(
            nn.Conv2d(                  # -> indim x 28 x 28
                in_channels = in_dim,   # 图片“厚度”，灰度图像为一维，RGB为三维
                out_channels = 8,       # 多少个卷积层，卷积后的“厚度”
                kernel_size = 3,        # 卷积窗口大小， 3 x 3
                stride = 1,             # 步长为 1
                padding = 1             # 为维持卷积后图片大小不变
            ),                          # -> 8 x 28 x28
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2         # 池化降采样，2 x 2 窗口中选出最大值，避免数据量爆炸
            ),                          # -> 6 x 14 x 14
            nn.Conv2d(
                in_channels = 8,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 0
            ),                          # -> 16 x 12 x 12
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2
            )                           # -> 16 x 6 x 6
        )

        # fully connected layers 全连接层
        # 将前面过程计算得到的16 x 6 x 6 的图像特征输入神经网络进行分类
        self.fc = nn.Sequential(
            nn.Linear(16 * 6 * 6, 256),
            nn.Linear(256, 64),
            nn.Linear(64, out_class),
        )
    def forward(self, x):
        out = self.conv(x)
        # .view将多维数据展开为一维
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 构建CNN
# MNIST为灰度图像数据集维度只有 1 ，由于是 0 - 9 的数字，分类为 10
cnn = CNN(1, 10)

# 使用Adam优化器寻求最快的“下山方法”
optimizer = torch.optim.Adam(cnn.parameters(),lr = LR)
# 使用交叉熵函数作为损失函数
loss_func = nn.CrossEntropyLoss()

# train
for epoch in range(EPOCH):

    for step, data in enumerate(train_loader):
        # BATCH_SIZE = 100,因此一批有100张图片被读取进来
        # img 是这一批（100张）图片 label 是这一批（100张）图片对应的标签
        img, label = data
        output = cnn(img)
        # 计算损失
        loss = loss_func(output, label)
        # 由于backward()误差反向传播时梯度会累加而不是替换，因此每个BATCH将其置零
        optimizer.zero_grad()
        loss.backward()
        # 更新参数
        optimizer.step()

        # 每100个BATCH输出一次损失和精确度
        if step % 100 == 0:
            pred = torch.max(output, 1)[1].data.numpy()
            accuracy = float((pred == label.data.numpy()).astype(int).sum()) / float(label.size(0))
            print('Epoch:',epoch,'|| Loss:%.4f' % loss,'|| Accuracy:%.2f' % accuracy)

print('training end')

# test
for step, data in enumerate(test_loader):
    img, label = data
    output = cnn(img)
    if step % 20 == 0:
        test_pred = torch.max(output, 1)[1].data.numpy()
        print('Prediction number:', test_pred, '\nReal number:', label.numpy())
        accuracy = float((test_pred == label.data.numpy()).astype(int).sum()) / float(label.size(0))
        print('Accuracy:%.2f' % accuracy)
