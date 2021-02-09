print('Load modules...')
import foolbox
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import  L2DeepFoolAttack, LinfBasicIterativeAttack, FGSM, L2CarliniWagnerAttack, FGM, PGD
import torch
from tqdm import tqdm
from collections import OrderedDict
from models.vgg_cif10 import VGG
from models.vgg import vgg16_bn
import argparse

#processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--attack", default='fgsm', help="the attack method you want to use in order to create adversarial examples. Either fgsm, bim, pgd, df or cw")
parser.add_argument("--net", default='cif10', help="the dataset the net was trained on, einether cif10 or cif100")
args = parser.parse_args()
#choose attack
attack_method = args.attack
net = args.net

#load model
print('Load model...')
if net == 'cif10':
    model = VGG('VGG16')
    checkpoint = torch.load('./models/vgg_cif10.pth')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
elif net == 'cif100':
    model = vgg16_bn()
    model.load_state_dict(torch.load('./models/vgg_cif100.pth'))
    preprocessing = dict(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761], axis=-3)
model = model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#normalizing for cif10

fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

#load correctly classified data
testset = torch.load('./data/clean_data_'+net)

#setup depending on attack
if attack_method == 'fgsm':
    attack = FGSM()
    epsilons = [0.03]
elif attack_method == 'bim':
    attack = LinfBasicIterativeAttack()
    epsilons = [0.03]
elif attack_method == 'pgd':
    attack = PGD()
    epsilons = [0.03]
elif attack_method == 'df':
    attack = L2DeepFoolAttack()
    epsilons = None
elif attack_method == 'df':
    attack = L2CarliniWagnerAttack(steps=1000)
    epsilons = None
else:
    print('unknown attack')
    
images = []
images_advs = []
success_counter = 0
'Perform attacks...'    
for i in tqdm(range(len(testset))):
        image, label = testset[i]
        image = image.cuda()
        label = label.cuda()
        _, adv, success = attack(fmodel, image, criterion=foolbox.criteria.Misclassification(label),epsilons=epsilons)
        adv = adv[0] #list to tensor
        success = success[0]
        if success:
            images_advs.append(adv.squeeze_(0))
            images.append(image.squeeze_(0))
            success_counter +=1
            
print('attack success rate:',success_counter/len(testset))

torch.save(images,'./data/'+net+'_adversarial_images/'+net+'_images_'+attack_method)
torch.save(images_advs,'./data/'+net+'_adversarial_images/'+net+'_images_adv_'+attack_method)
print('Done performing attacks and adversarial examples are saved!')