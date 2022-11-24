#!/bin/sh

attack=$1
dataset=$2

if [ "$attack" = "BppAttack" ]
then 
    python -u models/$attack/bppattack.py --dataset $dataset --attack_mode all2one --squeeze_num 32 
elif [ "$attack" = "WaNet" ]
then
    python models/$attack/train.py --dataset $dataset --attack_mode all2one
elif [ "$attack" = "BadNet" ]
then
    python models/$attack/train.py --dataset $dataset --attack_mode all2one    
fi
