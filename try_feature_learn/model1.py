import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 32, 5, stride=2)
        self.fc1 = nn.Conv2d(32, 16, 5)
        self.fc2 = nn.Conv2d(16, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        # exit(0)
        # x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape)
        # exit(0)
        # x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.pool(self.fc1(x)))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        # print(x.shape)
        # exit(0)
        x = self.sigmoid(x)
        if self.training:
            x = x.view(-1)
        # print(x.shape)
        return x

if __name__ == '__main__':
    from torchsummary import summary

    model = Net()
    # summary(model, input_size=(3, 480, 640), device='cpu')
    summary(model, input_size=(3, 128, 128), device='cpu')
    torch.save(model.state_dict(), '_orig.pt')
