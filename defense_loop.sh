#!/bin/sh

device=$4
epoch=20
cur_dir=$PWD

declare -a datasetArray=("cifar10" "mnist")
declare -a attackArray=("BadNet" "WaNet" "BppAttack")
declear -a defenseArray=("neural_cleanse" "STRIP" "fine_pruning" "CLP")


for dataset in ${datasetArray[@]}; do
  for attck in ${attackArray[@]}; do
    for defense in ${defenseArray[@]}; do
      echo "dataset:$dataset attack:$attack defense:$defense"
      log_file=logs/defense_$attack_$defense_$dataset
      if [ -f "$log_file"]; then
        echo "log file exist"
      else
        nohup sh defense2.sh $attack $defense $device $epoch &> logs/defense_$attack_$defense_$dataset &
      fi
    done
  done
done
