# SpectralAdversarialDefense

This code belongs to the paper: https://ieeexplore.ieee.org/document/9533442. If you use this repo, please cite:

```sh
@INPROCEEDINGS{9533442,
  author={Harder, Paula and Pfreundt, Franz-Josef and Keuper, Margret and Keuper, Janis},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)}, 
  title={SpectralDefense: Detecting Adversarial Attacks on CNNs in the Fourier Domain}, 
  year={2021},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/IJCNN52387.2021.9533442}}
  ```

## How to run the code

Clone the repository and install the requirements
```sh
$ git clone https://github.com/paulaharder/SpectralAdversarialDefense.git
$ cd SpectralAdversarialDefense
$ conda env create -f requirements.yml
$ conda activate spectral_defense
```

There are two possiblities: Either use our data set with existing adversarial examples, in this case follow the instructions under 'Download' or generate the examples by yourself, by going threw 'Data generation'. For both possibilities conclude with 'Build a detector'.

### Download

Download the adversarial examples and their non-adversarial counterparts as well as the trained VGG-16 networks from:
https://cutt.ly/0jmLTm0 . Extract the folders for the adversarial examples into /data and the models in the main directory. Afterwards continue with 'Build detector'.

#### Data download

To get the data directly on your server use wget. For the adversarial examples based on CIFAR-10 use:
```sh
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rfSSXNKcquD03lLBXd8IskoZAmDyjzPL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rfSSXNKcquD03lLBXd8IskoZAmDyjzPL" -O cif10_adversarial_images.zip && rm -rf /tmp/cookies.txt
```

then unzip
```sh
$ unzip -o cif10_adversarial_images.zip -d data/
$ rm cif10_adversarial_images.zip 
```

For adversarial examples based on CIFAR-100 use:
```sh
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NeWUvU63F04aO8k285PJNnfrB21RoI91' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NeWUvU63F04aO8k285PJNnfrB21RoI91" -O cif100_adversarial_images.zip && rm -rf /tmp/cookies.txt
```

#### Model download

To get the weights for the VGG-16 netwroks for CIFAR-10 and CIFAR-100 run:

```sh
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1b4vyiNIghGV9nwMnMki5mpC6kujLHP11' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1b4vyiNIghGV9nwMnMki5mpC6kujLHP11" -O models.zip && rm -rf /tmp/cookies.txt
```

then unzip
```sh
$ unzip -o models.zip -d .
$ rm models.zip 
```

### Data generation

Train the VGG16 on CIFAR-10:
```sh
$ python train_cif10.py
```

or on CIFAR-100
```sh
$ python train_cif100.py
```

Copy the pth-files from the checkpoint directory to /models/vgg_cif10.pth or /models/vgg_cif100.pth. In detail: For cif10 go to the folder ./checkpoints; copy the file ckpt.pth to the folder ./models and rename it vgg_cif10.pth. For cif100: Go to the folder ./checkpoints/vgg16; select a folder; copy a file *.pth to the folder ./models and rename it vgg_cif100.pth.


The following skript will download the CIFAR-10/100 dataset and extract the CIFAR10/100 images, which are correctly classified by the network by running. Use --net cif10 for CIFAR-10 and --net cif100 for CIFAR-100
```sh
$ python generate_clean_data.py --net cif10
```

Then generate the adversarial examples, argument can be fgsm (Fast Gradient Sign Method), bim (Basic Iterative Method), pgd (Projected Gradient Descent), df (Deepfool), cw (Carlini and Wagner), :
```sh
$ python attack.py --attack fgsm
```

### Build detector

First extract the necessary characteristics to train a detector, choose a detector out of InputMFS, InputPFS, LayerMFS, LayerPFS, LID, Mahalanobis adn an attack argument as before: 

```sh
$ python extract_characteristics.py --attack fgsm --detector InputMFS
```


Then train a classifier on the characteristics for a specific attack and detector:
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

