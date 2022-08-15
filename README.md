# FedTransformer
This repo provides an evaluation framework for Federated Learning (FL) using Vision Transformer (ViT) and popular CNN models 
respectively.      

## Introduction
Recently, several efforts attempt to introduce ViT models into FL training. However, deploying and training such ViT models from scratch in practice is
not trivial, existing works overlook the existence of the clients with low resources (e.g., mobile phones), which is a
common and practical FL setting. To this end, we use low-resolution images as model input to satisfy the resource
constraints and investigate several ViT models to explore whether ViT models still outperform CNN models in this
setting. 

## Data Partitioning
We consider three partitioning methods, IID, Non-IID(4) and Non-IID(2), on Fashion MNIST and CIFAR10 datasets. The details 
are shown in the following:  

| Method  | Description  |
| :----------- |-----------:|
| IID  | the training images are uniformly distributed to each client  |
| Non-IID(4)  | each client at most owns 4 classes images  |
| Non-IID(2)  | each client at most owns 2 classes data  |

## Supported Models
### CNN Models
- [x] ResNet50
- [x] EfficientNet_B4
- [x] RegNetY_4G

### ViT Models
- [x] CCT
- [x] LeViT
 
## Experiment
To run an evaluation using ViT or CNN model on specified dataset, you should provide the following args in Python script:


| Args  | Description |
| :-----------: |-----------|
| model | model name, choose from ['resnet50', 'regnety_4g', 'efficient_b4", 'cct', 'levit']|
| dataset_name | choose from choose ['cifar10', 'fmnist']|
| data_dir | base directory of the data, such as '~/data/cifar10'|
| img_size | resolution of input image, 32 for CIFAR10, 28 for FMNIST|
| train_bs | training batch size in each client |
| partition | data partitioning method, choose from ['iid', 'noniid-4', 'noniid-2'] |
| max_round | max communication round, default is 100 |
| lr | learning rate of client trainer, default is 0.03 |
| wd | weight decay of client trainer, default is 0.0 |
| epochs | epoch number per round, default is 1 |
| gpu | gpu ids for training, such as 0 |

An example of training ResNet50 on Non-IID(4) partitioned CIFAR10 should be:  
```shell script
python main.py --model 'resnet50' \              
               --dataset_name 'cifar10' \        
               --data_dir '~/data/cifar10' \      
               --img_size 32 \                   
               --train_bs 256 \                  
               --partition 'noniid-4' \                
               --max_round 100 \                  
               --lr 0.03 \                          
               --wd 0.0 \
               --epochs 1 \
               --gpu 0                           
```
