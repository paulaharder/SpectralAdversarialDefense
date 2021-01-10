#this script extracts the correctly classified images
print('Load modules...')
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from models.vgg import VGG

#load model vgg16
print('Load model...')
model = VGG('VGG16')
checkpoint = torch.load('./models/checkpoint_vgg/ckpt.pth')
new_state_dict = OrderedDict()
for k, v in checkpoint['net'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model = model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#normalizing the data
print('Load CIFAR-10 test set')
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

testset_normalized = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader_normalized = torch.utils.data.DataLoader(testset_normalized, batch_size=1, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=2)

data_iter = iter(testloader)
clean_dataset = []
correct = 0
total = 0
i = 0
print('Classify images...')
for images, labels in testloader_normalized:
    data = data_iter.next()
    images=images.cuda()
    labels=labels.cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    if (predicted == labels):
        clean_dataset.append(data)
    i +=1
print('Accuracy of the network on the 1000 test images: %d %%' % (
    100 * correct / total))

torch.save(clean_dataset, './data/clean_data_cif')
print('Done extracting and saving correctly classified images!')