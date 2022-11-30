#!/bin/sh

attack=$1
defense=$2
dataset=$3
device=$4
epoch=$5
batch_size=128
cur_dir=$PWD
if [ "$defense" = 'neural_cleanse' ]
then
  cd defense/WaNet/defenses/neural_cleanse
  python -u neural_cleanse.py --dataset $dataset --attack $attack --attack_mode all2one --data_root "${cur_dir}/datasets/" --checkpoints "${cur_dir}/checkpoints/${attack}/" --result "${cur_dir}/defense_result/${defense}/" --device $device --bs $batch_size --epoch $epoch
  cd $cur_dir
fi