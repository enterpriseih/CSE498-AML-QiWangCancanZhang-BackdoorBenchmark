#!/bin/sh

attack=$1
dataset=$2
device=$3

if [ "$attack" = "BppAttack" ]
then 
    python -u attack/$attack/bppattack.py --dataset $dataset --attack_mode all2one --squeeze_num 32
elif [ "$attack" = "WaNet" ]
then
    python attack/$attack/train.py --dataset $dataset --attack_mode all2one --data_root 'datasets/' --checkpoints "checkpoints/${attack}/" --device $device
elif [ "$attack" = "BadNet" ]
then
    python attack/$attack/train.py --dataset $dataset --attack_mode all2one --data_root 'datasets/' --checkpoints "checkpoints/${attack}/" --device $device
fi
