import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, action_dim, max_action):
        super(Model, self).__init__()

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 4, 4, stride=2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(4)

        # self.dropout = nn.Dropout(0.5)

        self.lin2 = nn.Linear(432, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = torch.split(x, x.shape[2]//2, dim=2)[1]
        x = self.bn1(self.lr(self.conv1(x)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        # x = self.dropout(x)
        x = self.lin2(x)
        x[:, 0] = self.max_action * self.sigm(x[:, 0])  # because we don't want the duckie to go backwards
        x[:, 1] = self.tanh(x[:, 1])

        return x

if __name__ == '__main__':
    from torchsummary import summary

    model = Model(action_dim=2, max_action=1.0)
    summary(model, input_size=(3, 120, 160), device='cpu')
    torch.save(model.state_dict(), '_orig.pt')
    # print('------------------- quantized -------------------')
    # model = torch.quantization.quantize_dynamic(
    #     model, {nn.Linear}, dtype=torch.qint8
    # )
    # summary(model, input_size=(3, 120, 160), device='cpu')
    # torch.save(model.state_dict(), '_int8.pt')
