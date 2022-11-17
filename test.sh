#!/bin/sh

attack=$1
dataset=$2

if [ "$attack" = "WaNet" ]
then
    python models/Warping-based_Backdoor_Attack-release-main/eval.py --dataset $dataset --attack_mode all2one
fi