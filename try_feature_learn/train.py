# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as F
from tqdm import tqdm
from model1 import Net

class SquarePad:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image):
        w, h = image.size
        # max_wh = np.max([w, h])
        hp = int((self.image_size[1] - w) / 2)
        vp = int((self.image_size[0] - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')

if __name__ == '__main__':
    image_size = [128, 128]
    transform = transforms.Compose([
            SquarePad(image_size),
            transforms.CenterCrop(image_size),
            # transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = torchvision.datasets.ImageFolder('train', transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    net = Net()
    net.train()

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        torch.save(net.state_dict(), 'model.pt')
