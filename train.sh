#!/bin/sh

attack=$1
dataset=$2
device=$3
n_iters=$4
batch_size=128

if [ "$attack" = "BppAttack" ]
then 
    python -u attack/$attack/bppattack.py --dataset $dataset --attack_mode all2one --squeeze_num 32 --data_root 'datasets/' --checkpoints "checkpoints/${attack}/" --device $device --n_iters $n_iters --bs $batch_size
elif [ "$attack" = "WaNet" ]
then
    python attack/$attack/train.py --dataset $dataset --attack_mode all2one --data_root 'datasets/' --checkpoints "checkpoints/${attack}/" --device $device --n_iters $n_iters --bs $batch_size
elif [ "$attack" = "BadNet" ]
then
    python attack/$attack/train.py --dataset $dataset --attack_mode all2one --data_root 'datasets/' --checkpoints "checkpoints/${attack}/" --device $device --n_iters $n_iters --bs $batch_size
fi
