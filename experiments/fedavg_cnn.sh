#!/usr/bin/env bash

MODEL='resnet50'  # efficient_b4 regnety_4g
PART='iid'    # iid split-1 split-2
BS=256
MAX_R=100
GPU=0

DATA='fmnist'
DIR='/data/zxj/dataset/fmnist'

source activate zxj_single_vit

cd ../

### cifar10   iid 时 学习率可以高点
#python main.py --model 'resnet50' --train_bs $BS --partition 'iid' --max_round $MAX_R --gpu $GPU --client_num_in_total 10 --client_num_per_round 5 --lr 0.03
# 2317MiB

#python main.py --model 'efficient_b4' --train_bs $BS --partition 'split-2' --max_round $MAX_R --gpu $GPU --client_num_in_total 10 --client_num_per_round 5 --lr 0.03
# 3003MiB

#python main.py --model 'regnety_4g' --train_bs $BS --partition 'split-2' --max_round $MAX_R --gpu $GPU --client_num_in_total 10 --client_num_per_round 5 --lr 0.03
# 2129MiB


#### fmnist
# split-1
#python main.py --model 'resnet50' --train_bs $BS --partition 'split-1' --max_round $MAX_R --gpu $GPU --dataset_name $DATA --data_dir $DIR --img_size 28 --client_num_in_total 100 --client_num_per_round 10 --lr 0.03 --in_channels 3
# final: 2737MiB

# split-2
python main.py --model 'efficient_b4' --train_bs $BS --partition 'split-2' --max_round $MAX_R --gpu $GPU --dataset_name $DATA --data_dir $DIR --img_size 28 --client_num_in_total 100 --client_num_per_round 10 --lr 0.03 --in_channels 3

# iid: 差一个 regnety_4G
#python main.py --model 'regnety_4g' --train_bs $BS --partition 'iid' --max_round $MAX_R --gpu $GPU --dataset_name $DATA --data_dir $DIR --img_size 28 --client_num_in_total 100 --client_num_per_round 10 --lr 0.03 --in_channels 3




#python main.py --model 'efficient_b4' --train_bs $BS --partition 'split-2' --max_round $MAX_R --gpu $GPU --dataset_name $DATA --data_dir $DIR --img_size 28 --client_num_in_total 100 --client_num_per_round 10 --lr 0.03
# final: 3195MiB

#python main.py --model 'regnet_4g' --train_bs $BS --partition 'split-1' --max_round $MAX_R --gpu $GPU --dataset_name $DATA --data_dir $DIR --img_size 28 --client_num_in_total 100 --client_num_per_round 10 --lr 0.03 --in_channels 3
# final: 3039MiB