#!/bin/sh

attack=$1
dataset=$2
device=$3
n_iters=$4
batch_size=128

if [ "$attack" = "BppAttack" ]
then 
    python -u model/train/$attack.py --attack $attack --dataset $dataset --attack_mode all2one --squeeze_num 32 --data_root 'datasets/' --checkpoints "checkpoints/${attack}/" --device $device --n_iters $n_iters --bs $batch_size
elif [ "$attack" = "WaNet" ]
then
    python model/train/$attack.py --attack $attack --dataset $dataset --attack_mode all2one --data_root 'datasets/' --checkpoints "checkpoints/${attack}/" --device $device --n_iters $n_iters --bs $batch_size
elif [ "$attack" = "BadNet" ]
then
    python model/train/$attack.py --attack $attack --dataset $dataset --attack_mode all2one --data_root 'datasets/' --checkpoints "checkpoints/${attack}/" --device $device --n_iters $n_iters --bs $batch_size
elif [ "$attack" = "Blended" ]
then
    python model/train/$attack.py --attack $attack --dataset $dataset --attack_mode all2one --data_root 'datasets/' --checkpoints "checkpoints/${attack}/" --device $device --n_iters $n_iters --bs $batch_size
elif [ "$attack" = "ISSBA" ]
then
    python3.7 model/train/$attack.py --attack $attack --dataset $dataset --attack_mode all2one --data_root "datasets/" --checkpoints "checkpoints/${attack}/" --device $device --n_iters $n_iters


elif [ "$attack" = "Clean" ]
then
    python model/train/$attack.py --dataset $dataset --data_root 'datasets/' --checkpoints "checkpoints/${attack}/" --device $device --n_iters $n_iters --bs $batch_size
fi
