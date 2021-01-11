# SpectralAdversarialDefense


## How to run the code

Clone the repository and install the requirements
```sh
$ cd SpectralAdversarialDetection
$ pip install -r requirements.txt
```

### Data generation

First download the CIFAR-10 dataset and extract the CIFAR10 images, which are correctly classified by the network by running
```sh
$ python generate_clean_data.py
```

Then generate the adversarial examples, argument can be fgsm (Fast Gradient Sign Method), bim (Basic Iterative Method), pgd (Projected Gradient Descent), df (Deepfool), cw (Carlini and Wagner), :
```sh
$ python attack.py --attack fgsm
```

### Build a Detector

First extract the necessary characteristics to train a detector, choose a detector out of InputMFS, InputPFS, LayerMFS, LayerPFS, LID, Mahalanobis: 
```sh
$ python extract_characteristics.py --attack fgsm --detector InputMFS
```

Then train a LR classifier on the characteristics for a specific attack and characteristic:
```sh
$ python detect_adversarials.py --attack fgsm --detector InputMFS
```

## Other repositories used
For training the VGG-16 on CIFAR-10 we used:
https://github.com/kuangliu/pytorch-cifar
For generating the adversarial examples we used the toolbox foolbox:
https://github.com/bethgelab/foolbox
For the LID detector we used:
https://github.com/xingjunm/lid_adversarial_subspace_detection
For the Mahalanobis detector we used:
https://github.com/pokaxpoka/deep_Mahalanobis_detector

