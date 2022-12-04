import json
import os
import shutil
from time import time
import config
import numpy as np
import torch
import torch.nn.functional as F
from classifier_models import PreActResNet18, ResNet18
from networks.models import Denormalizer, NetC_MNIST, Normalizer
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar

import bchlib
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants


def ISSBA_preprocess(opt):
    model_path = opt.encorder_path #'ckpt'
    secret = 'a'
    secret_size = 100
    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
        'stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
        'residual'].name
    output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

    BCH_POLYNOMIAL = 137
    BCH_BITS = 5
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    data = bytearray(secret + ' ' * (7 - len(secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0, 0, 0, 0])
    return sess, output_stegastamp, output_residual, input_secret, input_image, secret

def ISSBA_encoder(image,**args):
    sess = args['sess']
    output_stegastamp = args['output_stegastamp']
    output_residual = args['output_residual']
    input_secret = args['input_secret']
    input_image = args['input_image']
    secret = args['secret']

    width = 224
    height = 224

    # original size
    ori_width, ori_height = image.shape[0], image.shape[1]

    image = (image * 255).astype(np.uint8)

    n_c = np.array(image).shape[2]
    if n_c == 1:
        image = np.reshape(image, [ori_width, ori_height])

        # resize
    image = Image.fromarray(image)
    image = image.resize([width, height]).convert('RGB')
    image = np.array(image, dtype=np.float32) / 255.
    '''
    image: np.array (224, 224, 3) int(0,255)
	output: np.array (224, 224, 3) int(0,255)
    '''

    feed_dict = {
        input_secret: [secret],
        input_image: [image]
    }
    hidden_img, residual = sess.run([output_stegastamp, output_residual], feed_dict=feed_dict)

    # take to orginal channels
    if n_c == 1:
        hidden_img = hidden_img[0, :, :, 0]
    else:
        hidden_img = hidden_img[0]
    hidden_img = (hidden_img * 255).astype(np.uint8)

    # resize
    hidden_img = Image.fromarray(hidden_img)
    hidden_img = hidden_img.resize([ori_width, ori_height])
    hidden_img = np.array(hidden_img, dtype=np.float32) / 255.
    if n_c == 1:
        hidden_img = np.reshape(hidden_img, [ori_width, ori_height, 1])
    return hidden_img


def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "celeba":
        netC = ResNet18().to(opt.device)
    if opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


def train(netC, optimizerC, schedulerC, train_dl, tf_writer, epoch, opt, **args):
    print(" Train:")
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    total_bd = 0
    total_clean_correct = 0
    total_bd_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_BCE = torch.nn.BCELoss()

    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]

        # Create backdoor data for badnet
        num_bd = int(bs * rate_bd)
        inputs_bd = inputs[:num_bd]

        # mv axis
        inputs_bd = torch.moveaxis(inputs_bd, 1, 3)

        # tensor to np
        inputs_bd = inputs_bd.detach().cpu().numpy()

        for i in range(inputs_bd.shape[0]):
            inputs_bd[i] = ISSBA_encoder(inputs_bd[i], args)

        # np to tensor
        inputs_bd = torch.from_numpy(inputs_bd).to(opt.device)

        # mv axis to original
        inputs_bd = torch.moveaxis(inputs_bd, -1, 1)

        if opt.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
        if opt.attack_mode == "all2all":
            targets_bd = torch.remainder(targets[:num_bd] + 1, opt.num_classes)

        total_inputs = torch.cat([inputs_bd, inputs[num_bd:]], dim=0)
        total_inputs = transforms(total_inputs)
        total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
        start = time()
        total_preds = netC(total_inputs)
        total_time += time() - start

        loss_ce = criterion_CE(total_preds, total_targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()

        total_clean += bs - num_bd
        total_bd += num_bd
        total_clean_correct += torch.sum(
            torch.argmax(total_preds[num_bd:], dim=1) == total_targets[num_bd:]
        )
        total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)

        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        avg_acc_bd = total_bd_correct * 100.0 / total_bd

        avg_loss_ce = total_loss_ce / total_sample

        progress_bar(
            batch_idx,
            len(train_dl),
            "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} ".format(avg_loss_ce, avg_acc_clean, avg_acc_bd),
        )

        # Save image for debugging
        if not batch_idx % 50:
            if not os.path.exists(opt.temps):
                os.makedirs(opt.temps)
            path = os.path.join(opt.temps, "backdoor_image.png")

        # Image for tensorboard
        if batch_idx == len(train_dl) - 2:
            residual = inputs_bd - inputs[:num_bd]
            batch_img = torch.cat([inputs[:num_bd], inputs_bd, total_inputs[:num_bd], residual], dim=2)
            batch_img = denormalizer(batch_img)
            batch_img = F.upsample(batch_img, scale_factor=(4, 4))

    # for tensorboard
    if not epoch % 1:
        tf_writer.add_scalars(
            "Clean Accuracy", {"Clean": avg_acc_clean, "Bd": avg_acc_bd}, epoch
        )

    schedulerC.step()


def eval(
        netC,
        optimizerC,
        schedulerC,
        test_dl,
        best_clean_acc,
        best_bd_acc,
        tf_writer,
        epoch,
        opt,
        **args
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_ae_loss = 0

    criterion_BCE = torch.nn.BCELoss()

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            inputs_bd = inputs

            # mv axis
            inputs_bd = torch.moveaxis(inputs_bd, 1, 3)

            # tensor to np
            inputs_bd = inputs_bd.detach().cpu().numpy()

            for i in range(inputs_bd.shape[0]):
                inputs_bd[i] = ISSBA_encoder(inputs_bd[i],args)

            # np to tensor
            inputs_bd = torch.from_numpy(inputs_bd).to(opt.device)

            # mv axis to original
            inputs_bd = torch.moveaxis(inputs_bd, -1, 1)

            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets + 1, opt.num_classes)
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample

            info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
                acc_clean, best_clean_acc, acc_bd, best_bd_acc
            )
            progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    if not epoch % 1:
        tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, epoch)

    # Save checkpoint
    if acc_clean > best_clean_acc or (acc_clean > best_clean_acc - 0.1 and acc_bd > best_bd_acc):
        print(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        state_dict = {
            "netC": netC.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc,
            "epoch_current": epoch,
        }
        torch.save(state_dict, opt.ckpt_path)
        with open(os.path.join(opt.ckpt_folder, "results.txt"), "w+") as f:
            results_dict = {
                "epoch": epoch,
                "clean_acc": best_clean_acc.item(),
                "bd_acc": best_bd_acc.item(),
            }
            json.dump(results_dict, f, indent=2)

    return best_clean_acc, best_bd_acc


def main():
    opt = config.get_arguments().parse_args()

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    if opt.continue_training:
        if os.path.exists(opt.ckpt_path):
            print("Continue training!!")
            state_dict = torch.load(opt.ckpt_path)
            netC.load_state_dict(state_dict["netC"])
            optimizerC.load_state_dict(state_dict["optimizerC"])
            schedulerC.load_state_dict(state_dict["schedulerC"])
            best_clean_acc = state_dict["best_clean_acc"]
            best_bd_acc = state_dict["best_bd_acc"]
            epoch_current = state_dict["epoch_current"]
            tf_writer = SummaryWriter(log_dir=opt.log_dir)
        else:
            print("Pretrained model doesnt exist")
            exit()
    else:
        print("Train from scratch!!!")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        epoch_current = 0

        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        os.makedirs(opt.log_dir)
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)
        tf_writer = SummaryWriter(log_dir=opt.log_dir)

    sess, output_stegastamp, output_residual, input_secret, input_image, secret = ISSBA_preprocess(opt)
    params = {'sess':sess,
              'output_stegastamp':output_stegastamp,
              'output_residual':output_residual,
              'input_secret':input_secret,
              'input_image':input_image,
              'secret':secret}
    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(netC, optimizerC, schedulerC, train_dl, tf_writer, epoch, opt, params)
        best_clean_acc, best_bd_acc = eval(
            netC,
            optimizerC,
            schedulerC,
            test_dl,
            best_clean_acc,
            best_bd_acc,
            tf_writer,
            epoch,
            opt,
            params
        )


if __name__ == "__main__":
    main()
