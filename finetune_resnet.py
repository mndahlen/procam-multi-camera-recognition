import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from PersonDataset import PersonDataset 
import torchvision.models as models

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel = 3
num_classes = 7
learning_rate = 1e-3
batch_size = 20
num_epochs = 3

# Load Data
transform = transforms.Compose((
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
    transforms.Resize((224, 224)),
    transforms.ToTensor()))

dataset = PersonDataset(
    "data\hallway_639\data_labels.csv", 
    "data\hallway_639\persons", 
    transform=transform)
train_set, test_set = torch.utils.data.random_split(dataset, [339,300],
                      generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(dataset=train_set, batch_size=batch_size)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
resnet = models.resnet18(pretrained=True)
resnet.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)

if 1 :
    # Train network
    for epoch in range(1,num_epochs+1):
        losses = []
        print("Epoch #{}".format(epoch))
        for batch_idx, (data, targets) in enumerate(train_loader):
            print("Batch index: {}".format(batch_idx))
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Forward
            scores = resnet(data)
            loss = criterion(scores, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient decent
            optimizer.step()

torch.save(resnet, "models/resnet_hallway_639_{}_{}.tar".format(num_epochs,batch_size))


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    
        print("{}/{} correct ({})%".format(num_correct, num_samples, 100*num_correct/num_samples))
    model.train()

print("WARNING DISABLED")
if 1:
    check_accuracy(train_loader, resnet)
    check_accuracy(test_loader, resnet)