print('Load modules...')
import numpy as np
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from scipy.spatial.distance import cdist
from tqdm import tqdm
from collections import OrderedDict
from models.vgg_cif10 import VGG
from models.vgg import vgg16_bn
import argparse
import sklearn
import sklearn.covariance
    

#processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--attack", default='fgsm', help="the attack method which created the adversarial examples you want to use. Either fgsm, bim, pgd, df or cw")
parser.add_argument("--detector", default='InputMFS', help="the detector youz want to use, out of InputMFS, InputPFS, LayerMFS, LayerPFS, LID, Mahalanobis")
parser.add_argument("--net", default='cif10', help="the network used for the attack, either cif10 or cif100")
args = parser.parse_args()
#choose attack
attack_method = args.attack
detector = args.detector
net = args.net

#load adversarials and their non-adversarial counterpart
print('Loading images and adversarial examples...')
images = torch.load('./data/'+net+'_adversarial_images/'+net+'_images_'+attack_method)
images_advs = torch.load('./data/'+net+'_adversarial_images/'+net+'_images_adv_'+attack_method)
number_images = len(images)

#load model vgg16
print('Loading model...')
if net == 'cif10':
    model = VGG('VGG16')
    checkpoint = torch.load('./models/vgg_cif10.pth')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
elif net == 'cif100':
    model = vgg16_bn()
    model.load_state_dict(torch.load('./models/vgg_cif100.pth'))
else:
    print('unknown model')
model = model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#get a list of all feature maps of all layers
model_features = model.features
def get_layer_feature_maps(X, layers):
    X_l = []
    for i in range(len(model_features)):
        X = model_features[i](X)
        if i in layers:
            Xc = torch.Tensor(X.cpu())
            X_l.append(Xc.cuda())
    return X_l

#normalizatio
def cifar_normalize(images):
    if net == 'cif10':
        images[:,0,:,:] = (images[:,0,:,:] - 0.4914)/0.2023
        images[:,1,:,:] = (images[:,1,:,:] - 0.4822)/0.1994
        images[:,2,:,:] = (images[:,2,:,:] - 0.4465)/0.2010
    elif net == 'cif100':
        images[:,0,:,:] = (images[:,0,:,:] - 0.5071)/0.2675
        images[:,1,:,:] = (images[:,1,:,:] - 0.4867)/0.2565
        images[:,2,:,:] = (images[:,2,:,:] - 0.4408)/0.2761
    return images

#indice of activation layers
act_layers= [2,5,9,12,16,19,22,26,29,32,36,39,42]
fourier_act_layers = [9,16,22,29,36,42]


################Sections for each different detector

#######Fourier section

def calculate_fourier_spectrum(im, typ='MFS'):
    im = im.float()
    im = im.cpu()
    im = im.data.numpy() #transorm to numpy
    fft = np.fft.fft2(im)
    if typ == 'MFS':
        fourier_spectrum = np.abs(fft)
    elif typ == 'PFS':
        fourier_spectrum = np.abs(np.angle(fft))
    if net == 'cif100' and (attack_method=='cw' or attack_method=='df'):
        fourier_spectrum *= 1/np.max(fourier_spectrum)
    return fourier_spectrum


def calculate_spectra(images, typ='MFS'):
    fs = []   
    for i in range(len(images)):
        image = images[i]
        fourier_image = calculate_fourier_spectrum(image, typ=typ)
        fs.append(fourier_image.flatten())
    return fs
    
###Fourier Input

print('Extracting ' + detector+' characteristic...')
if detector == 'InputMFS':
    
    mfs = calculate_spectra(images)
    mfs_advs = calculate_spectra(images_advs)
        
    characteristics       = np.asarray(mfs, dtype=np.float32)
    characteristics_adv   = np.asarray(mfs_advs, dtype=np.float32)
    
elif detector == 'InputPFS':
    
    pfs = calculate_spectra(images, typ='PFS')
    pfs_advs = calculate_spectra(images_advs, typ='PFS')
        
    characteristics       = np.asarray(pfs, dtype=np.float32)
    characteristics_adv   = np.asarray(pfs_advs, dtype=np.float32)
###Fourier Layer   
elif detector == 'LayerMFS':
    
    mfs = []
    mfs_advs = []
    if net == 'cif100' and (attack_method=='cw' or attack_method=='df'):
        layers = [42]
    else:
        layers = fourier_act_layers
    for i in tqdm(range(number_images)):
        image = images[i].unsqueeze_(0)
        adv = images_advs[i].unsqueeze_(0)
        image = cifar_normalize(image)
        adv = cifar_normalize(adv)
        image_feature_maps = get_layer_feature_maps(image, layers) 
        adv_feature_maps = get_layer_feature_maps(adv, layers) 
        fourier_maps = calculate_spectra(image_feature_maps)
        fourier_maps_adv = calculate_spectra(adv_feature_maps)
        mfs.append(np.hstack(fourier_maps))
        mfs_advs.append(np.hstack(fourier_maps_adv))
        
    characteristics       = np.asarray(mfs, dtype=np.float32)
    characteristics_adv   = np.asarray(mfs_advs, dtype=np.float32)
    
elif detector == 'LayerPFS':

    pfs = []
    pfs_advs = []
    if net == 'cif100' and (attack_method=='cw' or attack_method=='df'):
        layers = [42]
    else:
        layers = fourier_act_layers
    for i in tqdm(range(number_images)):
        image = images[i].unsqueeze_(0)
        adv = images_advs[i].unsqueeze_(0)
        image = cifar_normalize(image)
        adv = cifar_normalize(adv)
        image_feature_maps = get_layer_feature_maps(image, layers) 
        adv_feature_maps = get_layer_feature_maps(adv, layers) 
        fourier_maps = calculate_spectra(image_feature_maps, typ='PFS')
        fourier_maps_adv = calculate_spectra(adv_feature_maps, typ='PFS')
        pfs.append(np.hstack(fourier_maps))
        pfs_advs.append(np.hstack(fourier_maps_adv))
        
    characteristics       = np.asarray(pfs, dtype=np.float32)
    characteristics_adv   = np.asarray(pfs_advs, dtype=np.float32)

#######LID section
elif detector == 'LID':

    #hyperparameters
    batch_size = 100
    if net == 'cif10':
        k = 20
    else:
        k = 10
    
    def mle_batch(data, batch, k):
        data = np.asarray(data, dtype=np.float32)
        batch = np.asarray(batch, dtype=np.float32)
        k = min(k, len(data)-1)
        f = lambda v: - k / np.sum(np.log(v/v[-1]))
        a = cdist(batch, data)
        a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
        a = np.apply_along_axis(f, axis=1, arr=a)
        return a

    lid_dim = len(act_layers)
    shape = np.shape(images[0])
    
    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(images), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch       = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_adv   = np.zeros(shape=(n_feed, lid_dim))
        batch= torch.Tensor(n_feed, shape[0], shape[1], shape[2])
        batch_adv= torch.Tensor(n_feed, shape[0], shape[1], shape[2])
        for j in range(n_feed):
            batch[j,:,:,:] = images[j]
            batch_adv[j,:,:,:] = images_advs[j]
        batch = cifar_normalize(batch)
        batch_adv = cifar_normalize(batch_adv)
        X_act       = get_layer_feature_maps(batch.to(device), act_layers)
        X_adv_act   = get_layer_feature_maps(batch_adv.to(device), act_layers)
        for i in range(lid_dim):
            X_act[i]       = np.asarray(X_act[i].cpu().detach().numpy()    , dtype=np.float32).reshape((n_feed, -1))
            X_adv_act[i]   = np.asarray(X_adv_act[i].cpu().detach().numpy()  , dtype=np.float32).reshape((n_feed, -1))
            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            lid_batch[:, i]       = mle_batch(X_act[i], X_act[i]      , k=k)
            lid_batch_adv[:, i]   = mle_batch(X_act[i], X_adv_act[i]  , k=k)
        return lid_batch, lid_batch_adv

    lids = []
    lids_adv = []
    n_batches = int(np.ceil(len(images) / float(batch_size)))
    
    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_adv = estimate(i_batch)
        lids.extend(lid_batch)
        lids_adv.extend(lid_batch_adv)

    characteristics       = np.asarray(lids, dtype=np.float32)
    characteristics_adv   = np.asarray(lids_adv, dtype=np.float32)

#######Mahalanobis section
elif detector == 'Mahalanobis':

    is_sample_mean_calculated = False #set true if sample mean and precision are already calculated
    
    if not is_sample_mean_calculated:
        print('Calculate sample mean and precision for Mahalanobis...')
        #load cifar10 training data
        if net == 'cif10':
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
            num_classes = 10
        elif net == 'cif100':
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
            num_classes = 100

        data_iter = iter(trainloader)
        im = data_iter.next()
        feature_list=[]
        layers = get_layer_feature_maps(im[0].cuda(), act_layers)
        m = len(act_layers)
        
        for i in tqdm(range(m)):
            layer = layers[i]
            n_channels=layer.shape[1]
            feature_list.append(n_channels)
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        correct, total = 0, 0
        num_output = len(feature_list)
        num_sample_per_class = np.empty(num_classes)
        num_sample_per_class.fill(0)
        list_features = []
        for i in range(num_output):
            temp_list = []
            for j in range(num_classes):
                temp_list.append(0)
            list_features.append(temp_list)
        for data, target in trainloader:
            total += data.size(0)
            data = data.cuda()
            with torch.no_grad():
                data = Variable(data)
                out_features = get_layer_feature_maps(data,act_layers)
            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)
            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = out[i].view(1, -1)
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        out_count += 1                
                num_sample_per_class[label] += 1
        sample_class_mean = []
        out_count = 0
        for num_feature in feature_list:
            temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
            for j in range(num_classes):
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            sample_class_mean.append(temp_list)
            out_count += 1
        precision = []
        for k in range(num_output):
            X = 0
            for i in range(num_classes):
                if i == 0:
                    X = list_features[k][i] - sample_class_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
            # find inverse            
            group_lasso.fit(X.cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().cuda()
            precision.append(temp_precision)
        np.save('./data/sample_mean_'+net,sample_class_mean)
        np.save('./data/precision_'+net,precision)
        
    #load sample mean and precision
    print('Loading sample mean and precision...')
    sample_mean = np.load('./data/sample_mean_'+net+'.npy',allow_pickle=True)
    precision =  np.load('./data/precision_'+net+'.npy',allow_pickle=True)    
    
    if net == 'cif10':
        if attack_method == 'fgsm':
            magnitude = 0.0002
        elif attack_method == 'cw':
            magnitude = 0.00001
        else:
            magnitude = 0.00005
    else:
        if attack_method == 'fgsm':
            magnitude = 0.005
        elif attack_method == 'cw':
            magnitude = 0.00001
        elif attack_method == 'df':
            magnitude = 0.0005
        else:
            magnitude = 0.01

    image_loader = torch.utils.data.DataLoader(images, batch_size=100, shuffle=True)
    adv_loader = torch.utils.data.DataLoader(images_advs, batch_size=100, shuffle=True)

    def get_mah(test_loader, layer_index):
        Mahalanobis = []
        for data in test_loader:
            data = cifar_normalize(data)
            data = data.cuda()
            data = Variable(data, requires_grad = True)
            out_features = get_layer_feature_maps(data, [act_layers[layer_index]])[0]
            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            out_features = torch.mean(out_features, 2)
            gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[layer_index][i]
                zero_f = out_features.cpu().data - batch_sample_mean.cpu()
                term_gau = -0.5*torch.mm(torch.mm(zero_f.cuda(),precision[layer_index].cuda()), zero_f.t().cuda()).diag()
                if i == 0:
                    gaussian_score = term_gau.view(-1,1)
                else:
                    gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
            # Input_processing
            sample_pred = gaussian_score.max(1)[1]
            batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
            zero_f = out_features - Variable(batch_sample_mean.cuda())
            pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index].cuda())), zero_f.t()).diag()
            loss = torch.mean(-pure_gau)
            loss.backward()
            gradient =  torch.ge(data.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            tempInputs = torch.add(data.data,  gradient, alpha=-magnitude)
            with torch.no_grad():
                noise_out_features = get_layer_feature_maps(Variable(tempInputs), [act_layers[layer_index]])[0]
                noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
                noise_out_features = torch.mean(noise_out_features, 2)
                noise_gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[layer_index][i]
                zero_f = noise_out_features.cpu().data - batch_sample_mean.cpu()
                term_gau = -0.5*torch.mm(torch.mm(zero_f.cuda(), precision[layer_index].cuda()), zero_f.t().cuda()).diag()
                if i == 0:
                    noise_gaussian_score = term_gau.view(-1,1)
                else:
                    noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      
            noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
            Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
        return Mahalanobis
    print('Calculating Mahalanobis scores...')
    Mah_adv = np.zeros((len(images_advs),len(act_layers)))
    Mah = np.zeros((len(images_advs),len(act_layers)))
    
    for layer_index in tqdm(range(len(act_layers))):
        Mah_adv[:,layer_index]=np.array(get_mah(adv_loader, layer_index))
        Mah[:,layer_index]=np.array(get_mah(image_loader, layer_index))
    
    characteristics = Mah
    characteristics_adv = Mah_adv
else:
    print('unknown detector')

###saving characteristics
np.save('./data/characteristics/'+net+'_'+attack_method+'_'+detector,characteristics)
np.save('./data/characteristics/'+net+'_'+attack_method+'_'+detector+'_adv',characteristics_adv)
print('Done extracting and saving characteristics!')