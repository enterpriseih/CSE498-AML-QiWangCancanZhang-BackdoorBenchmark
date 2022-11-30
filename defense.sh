#!/bin/sh

attack=$1
defense=$2
dataset=$3
device=$4
n_iters=$5
batch_size=128

if [ "$defense" = 'neural_cleanse' ]
then
  python -u defense/WaNet/defenses/neural_cleanse.py --dataset $dataset --attack $attack --attack_mode all2one --data_root 'datasets/' --checkpoints "checkpoints/${attack}/" --device $device
fi