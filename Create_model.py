import torch
import torch.nn as nn
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

model2 = torchvision.models.resnet18(pretrained=True)
for param in model2.parameters():
    param = param.requires_grad_(False)


# Add new layers to the models
model.fc = nn.Linear(model.fc.in_features, len(trainset.classes))
model2.fc = nn.Linear(model2.fc.in_features, len(trainset.classes))

print("The new fully connected layer for model1 is : ", model.fc)
print("The new fully connected layer for model2 is : ", model2.fc)

# Moving the models to GPU
model.cuda()
model2.cuda()

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
    model2.train()

    print("-----------------------------Training Mode-----------------------------------")
    print(f"-----------------------------EPOCH {epoch}-----------------------------------------")
    for batch_i, (images, target) in enumerate(trainloader):
        images = images.cuda()
        target = target.cuda()
        print("batch:", batch_i)

        # The models predicts the output
        output = model(images)
        output2 = model2(images)

        # Finding the loss between the real values and the output of the model
        loss1 = criterion(output, target)
        loss2 = criterion(output2, target)

        # BackPropagation
        optimizer.zero_grad()
        loss = loss1 + loss2
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

    print("-----------------------------Validation Mode---------------------------------")
    print(f"-----------------------------EPOCH {epoch}-----------------------------------------", epoch)
    for batch_i, (images, target) in enumerate(validloader):
        print("batch:", batch_i)
        images = images.cuda()
        target = target.cuda()
        output1 = model(images)
        output2 = model2(images)
        val_loss1 = criterion(output1, target).item()
        val_loss2 = criterion(output2, target).item()
        val_loss += val_loss1 + val_loss2

    # Save the models with the smallest val error
    if val_loss < best_epoch_loss:

        torch.save(model.state_dict(), "model1.pth")
        torch.save(model2.state_dict(), "model2.pth")

        best_model = epoch
        best_epoch_loss = val_loss
    # Save the current value of val loss to compare it to the next value of vall loss


print("The best combination of the model was at", best_model, "epoch")