#!/usr/bin/env bash

MODEL='levit'  # efficient_b4 regnety_4g
PART='iid'    # iid split-1 split-2
BS=256
MAX_R=100
GPU=0

DATA='fmnist'
DIR='/data/zxj/dataset/fmnist'

cd ../
source activate zxj_single_vit

#### cifar10
## iid
#python main.py --model 'levit' --train_bs $BS --partition 'iid' --max_round $MAX_R --gpu $GPU
# 13176: 1959MiB

#python main.py --model 'cct' --train_bs $BS --partition 'split-2' --max_round $MAX_R --gpu $GPU --client_num_in_total 10 --client_num_per_round 5 --lr 0.03 --client_optimizer sgd
# 2509MiB

#python main.py --model 'levit' --train_bs $BS --partition 'split-2' --max_round $MAX_R --gpu $GPU --client_num_in_total 10 --client_num_per_round 5 --lr 0.03 --client_optimizer sgd
# 1763MiB

#python main.py --model 't2t' --train_bs $BS --partition 'iid' --max_round $MAX_R --gpu $GPU

#### fmnist
python main.py --model 'levit' --train_bs $BS --partition 'split-2' --max_round $MAX_R --gpu $GPU --dataset_name $DATA --data_dir $DIR --img_size 28 --client_num_in_total 100 --client_num_per_round 10 --lr 0.03 --client_optimizer sgd --in_channels 3
# final: 2089MiB


#python main.py --model 'cct' --train_bs $BS --partition 'split-2' --max_round $MAX_R --gpu $GPU --dataset_name $DATA --data_dir $DIR --img_size 28 --client_num_in_total 100 --client_num_per_round 10 --lr 0.03 --client_optimizer sgd --in_channels 3
# final: 3000MiB  全部数据分布都结束