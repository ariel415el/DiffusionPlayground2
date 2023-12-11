import torch

from models.unet import Unet
from models.FC import FC


def human_format(num):
    """
    :param num: A number to print in a nice readable way.
    :return: A string representing this number in a readable way (e.g. 1000 --> 1K).
    """
    magnitude = 0

    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])  # add more suffices if you need them


def print_num_params(model):
    n = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    n += sum(p.nelement() for p in model.buffers() if p.requires_grad)
    return human_format(n)

if __name__ == "__main__":
    x = torch.randn(15, 3, 64, 64)
    t = torch.randint(0, 1000, (15,))
    unet = Unet(1000, 256, in_channels=3, out_channels=2, base_dim=32, dim_mults=[2, 4])
    print(print_num_params(unet))
    print(unet(x,t).shape)

    FC = FC(64, 1000, 256, in_channels=3, out_channels=2, bottleneck_dim=32, n_layers=2)
    print(print_num_params(FC))
    print(FC(x,t).shape)
