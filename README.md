# CIFAR10
### **various models adaption for CIFAR10 dataset**

## **Models**
- [x] Decision Tree
- [x] Random Forest
- [x] Logistic Regression
- [x] SVM (One-vs-Rest)
- [x] CNN (basic custom)
- [x] ResNet18 (scratch)
  - model/Resnet.py
- [x] ResNet50 (pretrained)
  - Backbone : ImagenetV2 
  - torchvision.models
- [x] ESRGAN
  - srgan/esrgan/models.py
- [ ] EfficientNet
- add more soon

## **Dataset**

* Information : https://www.cs.toronto.edu/~kriz/cifar.html
* shape: 
  * training : (50000, 32, 32, 3)
  * test : (10000, 32, 32, 3)
* class : 10
* example

![dataset](https://user-images.githubusercontent.com/50198431/205570183-67128e50-be8d-4e88-9d8e-94875d7c28f9.png)
* distribution

![distribution](https://user-images.githubusercontent.com/50198431/205570320-ef018b2f-956d-43c3-9dee-57b4f5c3198f.png)

## Data augmentation
- [x] RandomCrop
- [x] RandomHorizontalFlip
- [x] RandomVerticalFlip
- [x] RandomRotation
- [ ] super resolution
- [x] interpolation
* add more soon

## Result
* table will be here

## reference papers

* Ashish Vaswani, Attention is all you need, 2017(transformer)
  * https://arxiv.org/abs/1706.03762

* Lia deng, ImageNet: A Large-Scale Hierarchical Image Database, 2009
  * https://ieeexplore.ieee.org/document/5206848

* kaiming he, 2016, Deep Residual Learning for Image Recognition (ResNet)
  * https://arxiv.org/abs/1512.03385

* ryo takahashi, 2019,  Data Augmentation using Random Image Cropping and Patching for Deep CNNs (Random Image Cropping)
  * https://arxiv.org/abs/1906.11172

* Connor Shorten, 2019, A survey on Image Data Augmentation for Deep Learning (Data Augmentation - flip, rotate, crop, etc)
  * https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0

* Christian Ledig, 2016, Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN)
  * https://arxiv.org/abs/1609.04802

* Xintao Wang, 2018, ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN)
  * https://arxiv.org/abs/1809.00219
