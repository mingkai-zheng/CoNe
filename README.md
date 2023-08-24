# CoNe: Contrast Your Neighbours for Supervised Image Classification

This repository contains PyTorch evaluation code, training code and pretrained models for CoNe.

For details see [CoNe: Contrast Your Neighbours for Supervised Image Classification](https://arxiv.org/abs/2308.10761) by Mingkai Zheng, Shan You, Lang Huang, Xiu Su, Fei Wang, Chen Qian, Xiaogang Wang, and Chang Xu


## Reproducing

To run the code, you probably need to change the Dataset setting (dataset/imagenet.py), and Pytorch DDP setting (util/dist_init.py) for your own server environments.

The distributed training of this code is based on slurm environment, we have provided the training scrips in script/train.sh


We also provide the pretrained model for ResNet50 

|          |Arch | BatchSize | Epochs | Top-1 | Download  |
|----------|:----:|:---:|:---:|:---:|:---:|
|  CoNe | ResNet50 | 1024 | 100  | 78.7 % | [100ep-ResNet50-CoNe.tar](https://drive.google.com/file/d/1UCHRBtxTmGxsd3mbb_hVQjpVP4IrXFwJ/view?usp=sharing) |

If you want to test the pretained model, please download the weights from the link above, and move it to the checkpoints folder (create one if you don't have .checkpoints/ directory). The evaluation scripts also have been provided in script/train.sh


## Citation
If you find that CoNe interesting and help your research, please consider citing it:
```
@article{zheng2023cone,
  title={CoNe: Contrast Your Neighbours for Supervised Image Classification},
  author={Zheng, Mingkai and You, Shan and Huang, Lang and Su, Xiu and Wang, Fei and Qian, Chen and Wang, Xiaogang and Xu, Chang},
  journal={arXiv preprint arXiv:2308.10761},
  year={2023}
}
```
