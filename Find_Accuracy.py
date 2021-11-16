import torch
import torch.nn as nn
import torchvision.datasets
import matplotlib.pyplot as plt

testset = torchvision.datasets.ImageFolder("test", transform=torchvision.transforms.ToTensor())

# Create The models again
model = torchvision.models.resnet34(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(testset.classes))
model2 = torchvision.models.resnet18(pretrained=True)
model2.fc = nn.Linear(model2.fc.in_features, len(testset.classes))

# Set requires_grad_ = False because we don't want to train the already trained model
for param in model.parameters():
  param = param.requires_grad_(False)
for param in model2.parameters():
  param = param.requires_grad_(False)

# Moving the model to GPU
model.cuda()
model2.cuda()

# Load the model parameters
model.load_state_dict(torch.load("model1.pth"))
model2.load_state_dict(torch.load("model2.pth"))

# Turn the model to eval mode
model.eval()
model2.eval()

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

    output1 = model(images)
    output2 = model2(images)
    output = output1 + output2
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