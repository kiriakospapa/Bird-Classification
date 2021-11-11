import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import torchvision.datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

# Data augmentation
transforms = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.ColorJitter()
])

trainset = torchvision.datasets.ImageFolder("train", transform=transforms)
validset = torchvision.datasets.ImageFolder("valid", transform=transforms)
testset = torchvision.datasets.ImageFolder("test", transform=transforms)


# Define the batch size
batch_size = 4

# Define the loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

# Single batch
print("Size of trainingset : ", len(trainloader))
print("Size of validset    : ", len(validloader))
print("Size of testset     : ", len(testloader))

# Define the pretrained model and set required_grad = False as we don't want to train it
model = torchvision.models.resnet34(pretrained=True)
for param in model.parameters():
    param = param.requires_grad_(False)

# Add new layers to the model
model.fc = nn.Linear(model.fc.in_features, len(trainset.classes))

print("The new fully connected layer is : ", model.fc)

# Moving the model to GPU
model.cuda()

# Define the learning rate
lr = 0.001

# Define the criterion
criterion = nn.CrossEntropyLoss().cuda()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Define the number of epochs
EPOCHS = 6

correct = 0
save_loss_every_batches = 10
los_in_every_epoch = []

# Use cuda to train the model
model.cuda()
best_model = 1
# Train the last layer of the model
for epoch in range(EPOCHS):

    batch_loss = []
    batchloss = 0
    epoch_loss = 0
    model.train()
    for batch_i, (images, target) in enumerate(trainloader):
        images = images.cuda()
        target = target.cuda()

        # The model predicts the output
        output = model(images)

        # Finding the loss between the real values and the output of the model
        loss = criterion(output, target)

        # BackPropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batchloss += loss.item()


        if (batch_i % save_loss_every_batches) == 0:
            batch_loss.append(batchloss)
            batchloss = 0

    # Plot tha total batch loss for an epoch
    plt.plot(batch_loss)
    plt.axis([0, len(batch_loss), 0, max(batch_loss)])
    plt.show()

    # Validation
    model.eval()
    val_loss = 0
    best_epoch_loss = np.inf
    for batch_i, (images, target) in enumerate(validloader):
        images = images.cuda()
        target = target.cuda()
        output = model(images)
        val_loss += criterion(output, target).item()

    # Save the model with the smallest val error
    if val_loss < best_epoch_loss:

        torch.save(model.state_dict(), "pretrained_resnet.pth")
        best_model = epoch
        best_epoch_loss = val_loss
    # Save the current value of val loss to compare it to the next value of vall loss


print("The best model was at", best_model, "epoch")




