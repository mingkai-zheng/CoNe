#!/bin/bash

# ====== Training script ===============
# ./script/command.sh 100ep-resnet50-cone    8 1 "python -u cone_train.py        --use_fp16 --sync_bn  --bs 128 --epochs 100 --layers 1 --use_bn --backbone resnet50 --alpha_sup 0.7 --alpha_dc 0.4 --t_sup 0.10 --t_dc 0.07 --knn_k 512  --checkpoint 100ep-resnet50-cone.tar"


# ====== Evaluation script ===============
# ./script/command.sh 100ep-resnet50-cone    8 1 "python -u cone_train.py --eval --bs 128 --layers 1 --use_bn --backbone resnet50 --checkpoint 100ep-resnet50-cone.tar"

