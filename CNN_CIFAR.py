'''
coding time: May 14 2019
CIFAR数据集：
50K张训练影像
10K张测试影像
RGB三通道 影像分辨率：32 * 32
原理架构与MNIST数据集一致
注释参考CNN_MNIST
'''

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

# hyper parameters
EPOCH = 20
BATCH_SIZE = 100
LR = 1e-3

train_data = torchvision.datasets.CIFAR10(
    root = './Data/',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = False
)

test_data = torchvision.datasets.CIFAR10(
    root = './Data/',
    train = False,
    transform = torchvision.transforms.ToTensor(),
)


'''
in :print(train_data)
out:Dataset CIFAR10
    Number of datapoints: 50000
    Split: train
    Root Location: ./Data/
    Transforms (if any): ToTensor()
    Target Transforms (if any): None
    
in :print(train_data[1])
out:(tensor([[[0.6039, 0.4941, 0.4118,  ..., 0.3569, 0.3412, 0.3098],
         [0.5490, 0.5686, 0.4902,  ..., 0.3765, 0.3020, 0.2784],
         [0.5490, 0.5451, 0.4510,  ..., 0.3098, 0.2667, 0.2627],
         ...,
         [0.6863, 0.6118, 0.6039,  ..., 0.1647, 0.2392, 0.3647],
         [0.6471, 0.6118, 0.6235,  ..., 0.4039, 0.4824, 0.5137],
         [0.6392, 0.6196, 0.6392,  ..., 0.5608, 0.5608, 0.5608]],

        [[0.6941, 0.5373, 0.4078,  ..., 0.3725, 0.3529, 0.3176],
         [0.6275, 0.6000, 0.4902,  ..., 0.3882, 0.3137, 0.2863],
         [0.6078, 0.5725, 0.4510,  ..., 0.3216, 0.2745, 0.2706],
         ...,
         [0.6549, 0.6039, 0.6275,  ..., 0.1333, 0.2078, 0.3255],
         [0.6039, 0.5961, 0.6314,  ..., 0.3647, 0.4471, 0.4745],
         [0.5804, 0.5804, 0.6118,  ..., 0.5216, 0.5255, 0.5216]],

        [[0.7333, 0.5333, 0.3725,  ..., 0.2784, 0.2784, 0.2745],
         [0.6627, 0.6039, 0.4627,  ..., 0.3059, 0.2431, 0.2392],
         [0.6431, 0.5843, 0.4392,  ..., 0.2510, 0.2157, 0.2157],
         ...,
         [0.6510, 0.6275, 0.6667,  ..., 0.1412, 0.2235, 0.3569],
         [0.5020, 0.5098, 0.5569,  ..., 0.3765, 0.4706, 0.5137],
         [0.4706, 0.4784, 0.5216,  ..., 0.5451, 0.5569, 0.5647]]]), 9) <- 9是label
         可以看出来Totensor 真的是十分的贴心 数据已经整理的整整齐齐
'''

train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    shuffle = True
)

test_loader = Data.DataLoader(
    dataset = test_data,
    batch_size = BATCH_SIZE,
    shuffle = False
)

class CNN(nn.Module):
    def __init__(self, in_dim, out_class):
        super(CNN,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels = in_dim,           #    3 x 32 x 32
                out_channels = 16,              #
                kernel_size = 5,
                stride = 1,
                padding = 2                     # ->16 x 32 x 32
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2                 # ->16 x 16 x 16
            ),
            nn.Conv2d(
                in_channels = 16,
                out_channels = 48,
                kernel_size = 5,
                stride = 1,
                padding = 2                     # ->48 x 16 x 16
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2                 # ->48 x 8 x 8
            )
        )
        self.fc = nn.Sequential(
            nn.Linear(48 * 8 * 8, 512),
            nn.Linear(512, 96),
            nn.Linear(96, out_class)
        )
    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

cnn = CNN(3, 10)

optimizer = torch.optim.Adam(cnn.parameters(),lr = LR)
loss_func = nn.CrossEntropyLoss()

#train
for epoch in range(EPOCH):
    for step, data in enumerate(train_loader):
        img, label = data
        output = cnn(img)
        loss = loss_func(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            pred = torch.max(output, 1)[1].data.numpy()
            accuracy = float((pred == label.data.numpy()).astype(int).sum()) / float(label.size(0))
            print('Epoch:', epoch, '|| Loss:%.4f' % loss, '|| Accuracy:%.2f' % accuracy)
print('training end')

# test
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'house', 'ship', 'trunk']
for step, data in enumerate(test_loader):
    img, label = data
    output = cnn(img)
    loss = loss_func(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        pred = torch.max(output, 1)[1].data.numpy()
        print(
            'Prediction class:', list( map(lambda x:label_names[int(x)], pred[:10]) ),
            '\nReal class:', list( map(lambda x:label_names[int(x)],label.numpy()[:10]) )
        )
        accuracy = float((pred == label.data.numpy()).astype(int).sum()) / float(label.size(0))
        print('Accuracy:%.2f' % accuracy)
