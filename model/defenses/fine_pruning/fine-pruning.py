import torch
import os
import torch.nn as nn
import copy
from config import get_arguments
import numpy as np
import torch.nn.functional as F

import sys
sys.path.append("../..")
sys.path.append("..")
from process import create_backdoor
from utils.dataloader import get_dataloader
from utils.utils import progress_bar
from classifier_models import PreActResNet18
from networks.models import Normalizer, Denormalizer, NetC_MNIST

def eval(netC, test_dl, opt, **args):
    print(" Eval:")
    if opt.attack == "WaNet":
        identity_grid = args["identity_grid"]
        noise_grid = args["noise_grid"]
    else:
        identity_grid = None
        noise_grid = None

    acc_clean = 0.0
    acc_bd = 0.0
    total_sample = 0
    total_correct_clean = 0
    total_correct_bd = 0

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        total_sample += bs

        # Evaluating clean
        preds_clean = netC(inputs)
        correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets)
        total_correct_clean += correct_clean
        acc_clean = total_correct_clean * 100.0 / total_sample

        # Evaluating backdoor
        if opt.attack == 'WaNet':
            inputs_bd = create_backdoor(inputs, opt, identity_grid=identity_grid, noise_grid=noise_grid)
        else:
            inputs_bd = create_backdoor(inputs, opt)

        targets_bd = torch.ones_like(targets) * opt.target_label
        preds_bd = netC(inputs_bd)
        correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
        total_correct_bd += correct_bd
        acc_bd = total_correct_bd * 100.0 / total_sample

        progress_bar(batch_idx, len(test_dl), "Acc Clean: {:.3f} | Acc Bd: {:.3f}".format(acc_clean, acc_bd))
    return acc_clean, acc_bd


def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    # number of classes
    if opt.dataset == "cifar10":
        opt.num_classes = 10
    elif opt.dataset == "mnist":
        opt.num_classes = 10
    else:
        raise Exception("Invalid Dataset")

    # image height and width
    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    else:
        raise Exception("Invalid Dataset")

    # Load models
    if opt.dataset == "cifar10":
        netC = PreActResNet18().to(opt.device)
    elif opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)
    else:
        raise Exception("Invalid dataset")

    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    print('ckpt_folder', opt.ckpt_folder)
    if os.path.exists(os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, opt.attack_mode))):
        opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, opt.attack_mode))
    elif os.path.exists(os.path.join(opt.ckpt_folder, "{}_{}.pth.tar".format(opt.dataset, opt.attack_mode))):
        opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}.pth.tar".format(opt.dataset, opt.attack_mode))
    else:
        raise Exception("checkpoint path not right, please check")

    state_dict = torch.load(opt.ckpt_path)
    if "netC" in state_dict:
        netC.load_state_dict(state_dict["netC"])
    elif "model" in state_dict:
        netC.load_state_dict(state_dict["model"])
    else:
        raise Exception("model not in state_dict, please check the model key in checkpoint")
    if opt.attack == "WaNet":
        identity_grid = state_dict["identity_grid"].to(opt.device)
        noise_grid = state_dict["noise_grid"].to(opt.device)
    netC.requires_grad_(False)
    netC.eval()
    netC.to(opt.device)
    print(state_dict["best_clean_acc"], state_dict["best_bd_acc"])

    # Prepare dataloader
    test_dl = get_dataloader(opt, train=False)

    for name, module in netC._modules.items():
        print(name)

    # Forward hook for getting layer's output
    container = []

    def forward_hook(module, input, output):
        container.append(output)

    if opt.dataset == "cifar10":
        hook = netC.layer4.register_forward_hook(forward_hook)
    elif opt.dataset == 'mnist':
        hook = netC.layer3.register_forward_hook(forward_hook)
    else:
        raise Exception("Invalid Dataset")

    # Forwarding all the validation set
    print("Forwarding all the validation dataset:")
    for batch_idx, (inputs, _) in enumerate(test_dl):
        inputs = inputs.to(opt.device)
        netC(inputs)
        progress_bar(batch_idx, len(test_dl))

    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    hook.remove()

    # Pruning times - no-tuning after pruning a channel!!!
    acc_clean = []
    acc_bd = []

    result_dir = opt.results
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, opt.attack_mode)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = os.path.join(result_path, "{}_{}_output.txt".format(opt.dataset, opt.attack_mode))
    print ('result_path', result_path)

    with open(result_path, "w") as outs:
        for index in range(pruning_mask.shape[0]):
            net_pruned = copy.deepcopy(netC)
            num_pruned = index
            if index:
                channel = seq_sort[index - 1]
                pruning_mask[channel] = False
            print("Pruned {} filters".format(num_pruned))
            if opt.dataset == "cifar10":
                net_pruned.layer4[1].conv2 = nn.Conv2d(
                    pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
                )
                net_pruned.linear = nn.Linear(pruning_mask.shape[0] - num_pruned, 10)

                # Re-assigning weight to the pruned net
                for name, module in net_pruned._modules.items():
                    if "layer4" in name:
                        module[1].conv2.weight.data = netC.layer4[1].conv2.weight.data[pruning_mask]
                        module[1].ind = pruning_mask
                    elif "linear" == name:
                        module.weight.data = netC.linear.weight.data[:, pruning_mask]
                        module.bias.data = netC.linear.bias.data
                    else:
                        continue
            elif opt.dataset == 'mnist':
                net_pruned.layer3.conv1 = nn.Conv2d(
                    pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=2, padding=1, bias=False
                )
                net_pruned.linear6 = nn.Linear((pruning_mask.shape[0] - num_pruned) * 16, 512)

                # Re-assigning weight to the pruned net
                for name, module in net_pruned._modules.items():
                    if "layer3" in name:
                        module.conv1.weight.data = netC.layer3.conv1.weight.data[pruning_mask]
                        module.ind = pruning_mask
                    elif "linear6" == name:
                        module.weight.data = netC.linear6.weight.data.reshape(-1, 64, 16)[:, pruning_mask].reshape(
                            512, -1
                        )  # [:, pruning_mask]
                        module.bias.data = netC.linear6.bias.data
                    else:
                        continue
            net_pruned.to(opt.device)
            if opt.attack == "WaNet":
                clean, bd = eval(net_pruned, test_dl, opt, identity_grid=identity_grid, noise_grid=noise_grid)
            else:
                clean, bd = eval(net_pruned, test_dl, opt)
            outs.write("%d %0.4f %0.4f\n" % (index, clean, bd))


if __name__ == "__main__":
    main()
