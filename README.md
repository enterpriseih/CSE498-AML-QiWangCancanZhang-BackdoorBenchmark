# AMLproject


mkdir data



## Attack
BppAttack 2022

WaNet 2021

BadNet 2016

1. train with dataset and save backdoored checkpoint
2. evaluate and save performance (BA, ASR)

## Defense
channel-Lipschitzness-based-pruning 2022

1. load backdoored checkpoint
2. revise checkpoint and save defensed checkpoint
3. evaluate and save performance (BA, ASR for defensed checkpoint)


## Current Progress
Attack:
BadNet ()
WaNet
BppAttack

Defense:
neural_cleanse (For all, do not need bad inputs)
STRIP (For BadNet, WaNet now)
FINE_PRUNING (PLAN: Friday)
CLP (PLAN: Friday)



# Modifications
1. made it can run in any cuda machine


## How to Attack
nohup sh train.sh Clean cifar10 'cuda:0' 50 &> train_clean &

nohup sh train.sh BppAttack cifar10 'cuda:1' 50 &> train_bppattack &

nohup sh train.sh WaNet cifar10 'cuda:2' 50 &> train_wanet &

nohup sh train.sh BadNet cifar10  'cuda:3' 50 &> train_badnet &

nohup sh train.sh Clean mnist 'cuda:0' 50 &> train_clean &

nohup sh train.sh BppAttack mnist 'cuda:1' 50 &> train_bppattack &

nohup sh train.sh WaNet mnist 'cuda:2' 50 &> train_wanet &

nohup sh train.sh BadNet mnist  'cuda:3' 50 &> train_badnet &


## How to Defense

nohup sh defense.sh BadNet neural_cleanse cifar10 'cuda:0' 20 &> defense_badnet_nc &
nohup sh defense.sh WaNet neural_cleanse cifar10 'cuda:0' 20 &> defense_wanet_nc &
nohup sh defense.sh BppAttack neural_cleanse cifar10 'cuda:0' 20 &> defense_bppattack_nc &

nohup sh defense.sh BadNet STRIP cifar10 'cuda:0' 20 &> defense_badnet_strip &
nohup sh defense.sh WaNet STRIP cifar10 'cuda:0' 20 &> defense_wanet_strip &

nohup sh defense.sh BadNet fine_pruning cifar10 'cuda:0' 20 &> defense_badnet_fine_pruning &
nohup sh defense.sh BadNet fine_pruning mnist 'cuda:0' 20 &> defense_badnet_fine_pruning_mnist &






neural_cleanse
modify detecting.py _get_classifier function, possibly need to modify the classifier's state dict key.

STRIP
STRIP.py create_backdoor function, need to add methods generating bad inputs for different methods














Notes: BadNet, Clean is written based on WaNet file (can be combined together in the future)
