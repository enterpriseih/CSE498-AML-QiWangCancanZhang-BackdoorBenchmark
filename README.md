# AMLproject




## Attack

BppAttack, WaNet, BadNet, ISSBA, Blended

## Defense

CLP, FinePrune, Neural Cleanse, STRIP, ShrinkPad


## How to Attack

sh train.sh attack_method dataset device epoch

For example:

sh train.sh Clean cifar10 'cuda:0' 50 

sh train.sh BppAttack cifar10 'cuda:1' 50 

## How to Defense

For example:

sh defense.sh attack_method defense_method dataset device epoch

sh defense.sh BadNet neural_cleanse cifar10 'cuda:0' 20
