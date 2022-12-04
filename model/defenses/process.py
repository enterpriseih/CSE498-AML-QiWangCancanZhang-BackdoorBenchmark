import torch
import torch.nn.functional as F
import numpy as np
from numba import jit
from numba.types import float64, int64
from PIL import Image
try:
    import bchlib
    import tensorflow as tf
    from tensorflow.python.saved_model import tag_constants
    from tensorflow.python.saved_model import signature_constants


def generate_blended_trigger(opt):
    trigger = Image.open(opt.blended_trigger_path).convert('RGB')
    trigger = trigger.resize((opt.input_height, opt.input_width))
    trigger = np.asarray(trigger) if opt.input_channel == 3 else np.asarray(trigger)[:, :, 0]
    trigger = torch.from_numpy(trigger).to(opt.device)
    trigger = torch.unsqueeze(torch.moveaxis(trigger, -1, 0), 0)
    return trigger

def create_backdoor(inputs, opt, **args):
    if opt.attack == 'WaNet':
        identity_grid = args['identity_grid']
        noise_grid = args['noise_grid']
        bs = inputs.shape[0]
        grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)
        bd_inputs = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
    elif opt.attack == 'BadNet':
        bd_inputs = inputs
        for i in range(1, 4):
            for j in range(1, 4):
                bd_inputs[:, :, i, j] = 255
    elif opt.attack == 'BppAttack':
        bd_inputs = back_to_np_4d(inputs, opt)
        if opt.dithering:
            for i in range(bd_inputs.shape[0]):
                bd_inputs[i, :, :, :] = torch.round(
                    torch.from_numpy(floydDitherspeed(bd_inputs[i].detach().cpu().numpy(), float(opt.squeeze_num))).to(
                        opt.device))
        else:
            inputs_bd = torch.round(bd_inputs / 255.0 * (opt.squeeze_num - 1)) / (opt.squeeze_num - 1) * 255
        bd_inputs = np_4d_to_tensor(inputs_bd, opt)
    elif opt.attack == 'Blended':
        trigger = generate_blended_trigger(opt)
        bd_inputs = (1 - opt.blended_rate) * inputs + opt.blended_rate * trigger

    elif opt.attack == 'ISSBA':
        sess, output_stegastamp, output_residual, input_secret, input_image, secret = ISSBA_preprocess(opt)
        params = {'sess': sess,
                  'output_stegastamp': output_stegastamp,
                  'output_residual': output_residual,
                  'input_secret': input_secret,
                  'input_image': input_image,
                  'secret': secret}

        bd_inputs = inputs
        bd_inputs = torch.moveaxis(bd_inputs, 1, 3)
        # tensor to np
        bd_inputs = bd_inputs.detach().cpu().numpy()

        for i in range(bd_inputs.shape[0]):
            bd_inputs[i] = ISSBA_encoder(bd_inputs[i], **params)

        # np to tensor
        bd_inputs = torch.from_numpy(bd_inputs).to(opt.device)

        # mv axis to original
        bd_inputs = torch.moveaxis(bd_inputs, -1, 1)
    else:
        raise Exception('attack name not right')
    return bd_inputs


# Following are process for BppAttack
def back_to_np(inputs, opt):
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif opt.dataset == "mnist":
        expected_values = [0.5]
        variance = [0.5]
    elif opt.dataset in ["gtsrb", "celeba"]:
        expected_values = [0, 0, 0]
        variance = [1, 1, 1]
    inputs_clone = inputs.clone()
    print(inputs_clone.shape)
    if opt.dataset == "mnist":
        inputs_clone[:, :, :] = inputs_clone[:, :, :] * variance[0] + expected_values[0]
    else:
        for channel in range(3):
            inputs_clone[channel, :, :] = inputs_clone[channel, :, :] * variance[channel] + expected_values[channel]
    return inputs_clone * 255


def back_to_np_4d(inputs, opt):
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif opt.dataset == "mnist":
        expected_values = [0.5]
        variance = [0.5]
    elif opt.dataset in ["gtsrb", "celeba"]:
        expected_values = [0, 0, 0]
        variance = [1, 1, 1]
    inputs_clone = inputs.clone()

    if opt.dataset == "mnist":
        inputs_clone[:, :, :, :] = inputs_clone[:, :, :, :] * variance[0] + expected_values[0]
    else:
        for channel in range(3):
            inputs_clone[:, channel, :, :] = inputs_clone[:, channel, :, :] * variance[channel] + expected_values[
                channel]

    return inputs_clone * 255


def np_4d_to_tensor(inputs, opt):
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif opt.dataset == "mnist":
        expected_values = [0.5]
        variance = [0.5]
    elif opt.dataset in ["gtsrb", "celeba"]:
        expected_values = [0, 0, 0]
        variance = [1, 1, 1]
    inputs_clone = inputs.clone().div(255.0)

    if opt.dataset == "mnist":
        inputs_clone[:, :, :, :] = (inputs_clone[:, :, :, :] - expected_values[0]).div(variance[0])
    else:
        for channel in range(3):
            inputs_clone[:, channel, :, :] = (inputs_clone[:, channel, :, :] - expected_values[channel]).div(
                variance[channel])
    return inputs_clone


@jit(float64[:](float64[:], int64, float64[:]), nopython=True)
def rnd1(x, decimals, out):
    return np.round_(x, decimals, out)


@jit(nopython=True)
def floydDitherspeed(image, squeeze_num):
    channel, h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[:, y, x]
            temp = np.empty_like(old).astype(np.float64)
            new = rnd1(old / 255.0 * (squeeze_num - 1), 0, temp) / (squeeze_num - 1) * 255
            error = old - new
            image[:, y, x] = new
            if x + 1 < w:
                image[:, y, x + 1] += error * 0.4375
            if (y + 1 < h) and (x + 1 < w):
                image[:, y + 1, x + 1] += error * 0.0625
            if y + 1 < h:
                image[:, y + 1, x] += error * 0.3125
            if (x - 1 >= 0) and (y + 1 < h):
                image[:, y + 1, x - 1] += error * 0.1875
    return image

# following are process for issba
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