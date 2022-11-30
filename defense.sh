#!/bin/sh

attack=$1
defense=$2
dataset=$3
device=$4
n_iters=$5
batch_size=128

if [ "$defense" = 'neural_cleanse' ]
then
  #cd defense/WaNet/defenses/neural_cleanse
  python -u defense/WaNet/defenses/neural_cleanse/neural_cleanse.py --dataset $dataset --attack $attack --attack_mode all2one --data_root 'datasets/' --checkpoints "checkpoints/${attack}/" --device $device --bs $batch_size --n_iters $n_iters
fi