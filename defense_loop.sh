#!/bin/bash

device=$1
epoch=20
cur_dir=$PWD

defense_list="conf/defense_methods"
attack_list="conf/attack_methods"
dataset_list=(mnist)

for dataset in ${dataset_list[@]}; do
    while IFS= read -r attack
    do
        while IFS= read -r defense
        do
            echo "dataset:$dataset attack:$attack defense:$defense"
            log_file="logs_defenses/defense_${attack}_${defense}_${dataset}"
            echo $log_file
            nohup sh defense2.sh $attack $defense $dataset $device $epoch &> $log_file &
        done < "$defense_list"
    done < "$attack_list"
done
