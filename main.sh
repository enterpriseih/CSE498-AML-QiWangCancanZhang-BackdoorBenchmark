#!/bin/sh

attack=$1
dataset=$2

attack_model_ckpt=${attack}_${dataset}
cd attack/$attack
if [ "$attack" = "BppAttack" ]
then 
    python -u models/BppAttack-main/bppattack.py --dataset $dataset --attack_mode all2one --squeeze_num 32 
elif [ "$attack" = "WaNet" ]
then
    python models/Warping-based_Backdoor_Attack-release-main/train.py --dataset $dataset --attack_mode all2one
elif [ "$attack" = "BadNet" ]
then
    python models/BadNet/train.py --dataset $dataset --attack_mode all2one --ckpt $attack_model_ckpt 
elif [ "$attack" = "Imperceptible" ]
then
    python main.py --dataset $dataset --num_class 43 --a 0.3 --b 0.1 --weight_decay 0
    python eval.py --dataset $dataset --num_class 43 > eval_${attack}_${dataset}_output
    python parse.py
fi
cd ..




