# SpectralAdversarialDefense


## How to run the code

Clone the repository and install the requirements
```sh
$ cd SpectralAdversarialDetection
$ pip install -r requirements.txt
```

There are two possiblities: Either use our data set with existing adversarial examples, in this case follow the instructions under 'Download' or generate the examples by yourself, by going threw 'Data generation'. For both possibilities conclude with 'Build a detector'.

### Download

Download the adversarial examples and their non-adversarial counterparts from:
https://cutt.ly/0jmLTm0 . Extract the folders in /data. Go to 'Build detector'.

### Data generation

Train the VGG16 on CIFAR-10/100:
```sh
$ python train_cif10.py
```

```sh
$ python train_cif100.py
```

Copy the pth-file in /models/vgg_cif10.py or /models/vgg_cif100.py.

Download the CIFAR-10/100 dataset and extract the CIFAR10/100 images, which are correctly classified by the network by running
```sh
$ python generate_clean_data.py --net cif10
```

Then generate the adversarial examples, argument can be fgsm (Fast Gradient Sign Method), bim (Basic Iterative Method), pgd (Projected Gradient Descent), df (Deepfool), cw (Carlini and Wagner), :
```sh
$ python attack.py --attack fgsm
```

### Build detector

First extract the necessary characteristics to train a detector, choose a detector out of InputMFS, InputPFS, LayerMFS, LayerPFS, LID, Mahalanobis: 

```sh
$ python extract_characteristics.py --attack fgsm --detector InputMFS
```


Then train a LR classifier on the characteristics for a specific attack and characteristic:
```sh
$ python detect_adversarials.py --attack fgsm --detector InputMFS
```

## Remark

To use the LayerMFS and LayerPFS detectors on CIFAR-100 for the Deepfool and C&W attacks rapids cuML needs to be installed:
https://rapids.ai/start.html .

## Other repositories used
* For training the VGG-16 on CIFAR-10 we used:
https://github.com/kuangliu/pytorch-cifar.
* For training on CIFAR-100:
https://github.com/weiaicunzai/pytorch-cifar100.
* For generating the adversarial examples we used the toolbox foolbox:
https://github.com/bethgelab/foolbox.
* For the LID detector we used:
https://github.com/xingjunm/lid_adversarial_subspace_detection.
* For the Mahalanobis detector we used:
https://github.com/pokaxpoka/deep_Mahalanobis_detector.

