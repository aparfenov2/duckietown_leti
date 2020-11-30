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
from train import SquarePad

image_size = [128, 128]
transform = transforms.Compose([
        SquarePad(image_size),
        transforms.CenterCrop(image_size),
        # transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

trainset = torchvision.datasets.ImageFolder('val', transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net()
state_dict = torch.load('model.pt', map_location=device)
model.load_state_dict(state_dict)

model.eval()

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

correct = 0
for i, data in enumerate(tqdm(trainloader)):
    inputs, labels = data
    outputs = model(inputs).round().long().squeeze()
    correct += (outputs == labels).float().sum()
    accuracy = 100 * correct / len(trainset)
print(accuracy)
