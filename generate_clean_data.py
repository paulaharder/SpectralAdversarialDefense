#this script extracts the correctly classified images
print('Load modules...')
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from models.vgg_cif10 import VGG
from models.vgg import vgg16_bn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--net", default='cif10', help="the network used for the attack, either cif10 or cif100")
args = parser.parse_args()
#choose attack
net = args.net


print('Load model...')

if net == 'cif10':
    model = VGG('VGG16')
    checkpoint = torch.load('./models/vgg_cif10.pth')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    

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
elif net == 'cif100':
    model = vgg16_bn()
    model.load_state_dict(torch.load('./models/vgg_cif100.pth'))
    print('Load CIFAR-100 test set')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    testset_normalized = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    testloader_normalized = torch.utils.data.DataLoader(testset_normalized, batch_size=1, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=2)
else:
    print('unknown net')
    
model = model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
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

torch.save(clean_dataset, './data/clean_data_'+net)
print('Done extracting and saving correctly classified images!')