import torch
import torch.nn.functional as F

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