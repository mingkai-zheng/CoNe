# CoNe: Contrast Your Neighbours for Supervised Image Classification

This repository contains PyTorch evaluation code, training code and pretrained models for CoNe.

## Reproducing

To run the code, you probably need to change the Dataset setting (dataset/imagenet.py), and Pytorch DDP setting (util/dist_init.py) for your own server enviroments.

The distributed training of this code is based on slurm environment, we have provided the training scrips in script/train.sh


We also provide the pretrained model for ResNet50 

|          |Arch | BatchSize | Epochs | Top-1 | Download  |
|----------|:----:|:---:|:---:|:---:|:---:|
|  CoNe | ResNet50 | 1024 | 100  | 78.7 % | [100ep-ResNet50-CoNe.tar](https://drive.google.com/file/d/1UCHRBtxTmGxsd3mbb_hVQjpVP4IrXFwJ/view?usp=sharing) |

If you want to test the pretained model, please download the weights from the link above, and move it to the checkpoints folder (create one if you don't have .checkpoints/ directory). The evaluation scripts also has been provided in script/train.sh

