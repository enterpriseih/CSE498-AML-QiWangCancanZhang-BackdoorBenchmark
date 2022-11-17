#!/bin/sh

attack=$1
dataset=$2

if [ "$attack" = "BppAttack" ]
then 
    python -u models/BppAttack-main/bppattack.py --dataset $dataset --attack_mode all2one --squeeze_num 32 
elif [ "$attack" = "WaNet" ]
then
    python models/Warping-based_Backdoor_Attack-release-main/train.py --dataset $dataset --attack_mode all2one
fi