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


## Task
1. Can: revise BppAttack, WaNet, and BadNet realize pass dasaset_path and checkpoint_path to attack train. 
2. attack eval: pass performance path_log to evaluate.




## How to Attack
nohup sh train.sh BppAttack cifar10 'cuda:1' 50 &> train_bppattack &

nohup sh train.sh WaNet cifar10 'cuda:2' 50 &> train_wanet &

sh test.sh WaNet cifar10

nohup sh train.sh BadNet cifar10  'cuda:3' 50 &> train_badnet &

Notes: BadNet is written based on WaNet file (can be combined together in the future)
