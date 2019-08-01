# AutoDeeplab

This is an implementation of [Auto-DeepLab](https://arxiv.org/abs/1901.02985) using Pytorch.

## Environment

The implementation needs the following dependencies:  

- Python = 3.7 

- Pytorch = 0.4 

- TensorboardX

Other basic dependencies like matplotlib, tqdm ... are also needed.

## Installation

First, clone the repository

    git clone https://github.com/MenghaoGuo/AutoDeeplab.git
    
Then

    cd AutoDeeplab

## Train

The dataloader module is built on this [repo](https://github.com/jfzhang95/pytorch-deeplab-xception)

If you want to train this model on different datasets, you need to edit --dataset parameter and then:

    bash train_voc.sh


## Reference
[1] : [Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation](https://arxiv.org/abs/1901.02985)


[2] : [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)
