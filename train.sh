#!/bin/sh

attack=$1
dataset=$2
device=$3
n_iters=$4
batch_size=128
cur_dir=$PWD

if [ "$attack" = "BppAttack" ]
then 
    python -u attack/$attack/bppattack.py --dataset $dataset --attack_mode all2one --squeeze_num 32 --data_root 'datasets/' --checkpoints "checkpoints/${attack}/" --device $device --n_iters $n_iters --bs $batch_size
elif [ "$attack" = "WaNet" ]
then
    python attack/$attack/train.py --dataset $dataset --attack_mode all2one --data_root 'datasets/' --checkpoints "checkpoints/${attack}/" --device $device --n_iters $n_iters --bs $batch_size
elif [ "$attack" = "BadNet" ]
then
    python attack/$attack/train.py --dataset $dataset --attack_mode all2one --data_root 'datasets/' --checkpoints "checkpoints/${attack}/" --device $device --n_iters $n_iters --bs $batch_size
elif [ "$attack" = "Clean" ]
then
    python attack/$attack/train.py --dataset $dataset --data_root 'datasets/' --checkpoints "checkpoints/${attack}/" --device $device --n_iters $n_iters --bs $batch_size
elif [ "$attack" = "ISSBA" ]
then
    cd attack/$attack
    python3.7 train.py --dataset $dataset --attack_mode all2one --data_root "${cur_dir}/datasets/" --checkpoints "${cur_dir}/checkpoints/${attack}/" --device $device --n_iters $n_iters --bs $batch_size
    cd $cur_dir
fi
