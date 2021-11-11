import torch
import torch.nn as nn
import torchvision.datasets
import matplotlib.pyplot as plt

testset = torchvision.datasets.ImageFolder("test", transform=torchvision.transforms.ToTensor())

# Create The model again
model = model = torchvision.models.resnet34(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(testset.classes))

# Set requires_grad_ = False because we don't want to train the already trained model
for param in model.parameters():
  param = param.requires_grad_(False)

# Moving the model to GPU
model.cuda()

# Load the model parameters
model.load_state_dict(torch.load("pretrained_resnet.pth"))

# Turn the model to eval mode
model.eval()

# Define the criterion
criterion = nn.CrossEntropyLoss().cuda()

# Define the batch size
batch_size = 4

# Create the testloader to iterate
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

# Initialize variables
running_loss = []
correct = 0
total = 0

for batch_idc, (images, target) in enumerate(testloader):
    images = images.cuda()
    target = target.cuda()
    model.eval()
    output = model(images)
    loss = criterion(output, target)
    running_loss.append(loss.item())
    _, predicted = output.max(1)
    correct += predicted.eq(target).sum().item()
    total += images.size(0)
    target.cpu()
    predicted.cpu()

print("The total accuracy is ", 100*(correct/total), "%")
plt.plot(running_loss)
plt.show()