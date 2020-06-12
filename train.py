import torch
import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt


my_trans = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.CenterCrop(224), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-45,45)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225))
 ])

#    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#    transforms.Normalize(mean=[0.4, 0.4, 0.4],std=[0.2, 0.2, 0.2])
#    transforms.RandomRotation((-45,45)),

    #    , #随机旋转  
    #    transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
    #transforms.RandomVerticalFlip(),
    
from torchvision import datasets

train_data = datasets.ImageFolder('./hamican/cancer/images_train', transform=my_trans)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)

test_data = datasets.ImageFolder('./hamican/cancer/images_test', transform=my_trans)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=True)

classes = ['0', '1', '2', '3']


import matplotlib.pyplot as plt
import numpy as np



def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# 首先是调用Variable、 torch.nn、torch.nn.functional
from torch.autograd import Variable  # 这一步还没有显式用到variable，但是现在写在这里也没问题，后面会用到
import torch.nn as nn
import torch.nn.functional as F


class CNNNet(nn.Module):  # 我们定义网络时一般是继承的torch.nn.Module创建新的子类
    def __init__(self):
        super(CNNNet, self).__init__()  # 第二、三行都是python类继承的基本操作,此写法应该是python2.7的继承格式,但python3里写这个好像也可以
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)       # 添加第一个卷积层,调用了nn里面的Conv2d（）
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层
        self.conv2 = nn.Conv2d(6, 16, 3)  # 同样是卷积层
        self.conv2_drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 54 * 54, 120)  # 接着三个全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # 这里定义前向传播的方法，为什么没有定义反向传播的方法呢？这其实就涉及到torch.autograd模块了，
        # 但说实话这部分网络定义的部分还没有用到autograd的知识，所以后面遇到了再讲
        x = self.pool(F.relu(self.conv1(x)))  # F是torch.nn.functional的别名，这里调用了relu函数 F.relu()
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        x = x.view(-1, 16 * 54 * 54)
        # 179776= 16*53*53*4
        #186624 =16*54*54*4 #4是批次batch，16是输入？什么。

        # x = x.view(-1, self.num_flat_features(x)) #为什么我要用这个？

        # x = x.view(-1, 16 * 5 * 5)  # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
        #  第一个参数-1是说这个参数由另一个参数确定， 比如矩阵在元素总数一定的情况下，确定列数就能确定行数。
        #  那么为什么这里只关心列数不关心行数呢，因为马上就要进入全连接层了，而全连接层说白了就是矩阵乘法，
        #  你会发现第一个全连接层的首参数是16*5*5，所以要保证能够相乘，在矩阵乘法之前就要把x调到正确的size
        # 更多的Tensor方法参考Tensor: http://pytorch.org/docs/0.3.0/tensors.html
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = CNNNet()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)

print(net)
import torch.optim as optim          #导入torch.potim模块

 # 和python中一样，类定义完之后实例化就很简单了，我们这里就实例化了一个net

criterion = nn.CrossEntropyLoss()    #同样是用到了神经网络工具箱 nn 中的交叉熵损失函数
# optimizer = torch.optim.Adam(net.parameters())
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)   #optim模块中的SGD梯度优化方式---随机梯度下降
from torch.utils.data import DataLoader

for epoch in range(4):  # loop over the dataset multiple times 指定训练一共要循环几个epoch
    running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
    for i, data in enumerate(train_loader, 0):  # 这里我们遇到了第一步中出现的trailoader，代码传入数据/#print(i)        #print(data)
        inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels，# get the inputs
        inputs, labels = inputs.to(device), labels.to(device)
        # inputs, labels = Variable(inputs), Variable(labels)
        # print(labels)
        # print(inputs) # forward + backward + optimize

        optimizer.zero_grad()  # 要把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度

        outputs = net(inputs)  # 把数据输进网络net，这个net()在第二步的代码最后一行我们已经定义了/# zero the parameter gradients
        # print(outputs)

        loss = criterion(outputs, labels)  # 计算损失值,criterion我们在第三步里面定义了
        loss.backward()  # loss进行反向传播，下文详解
        optimizer.step()  # 当执行反向传播之后，把优化器的参数进行更新，以便进行下一轮

        # print statistics                   # 这几行代码不是必须的，为了打印出loss方便我们看而已，不影响训练过程

        running_loss += loss.item()  # 从下面一行代码可以看出它是每循环0-199共两百次才打印一次
        if (i + 1) % 20 == 0:
            # if i % 50 == 49:    # print every 2000 mini-batches   所以每个200次之类先用running_loss进行累加
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))  # 然后再除以200，就得到这两千次的平均损失值
            running_loss = 0.0  # 这一个200次结束后，就把running_loss归零，下一个200次继续使用

print('Finished Training')
