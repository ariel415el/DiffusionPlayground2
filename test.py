import argparse
import json
import math
import os

import torch
from torchvision.utils import save_image

from DDPM import DDPM
from models import Unet

if __name__ == '__main__':
    device = torch.device('cuda:0')
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('model_dir', type=str)
    args = parser.parse_args()

    model_dir = args.model_dir
    args = json.load(open(os.path.join(model_dir, "args.txt")))

    in_channels = 1 if args['gray_scale'] else 3
    denoiser = Unet(args['timesteps'], 256, in_channels, in_channels, args['model_base_dim'], depth=args['denoiser_depth'])
    model = DDPM(denoiser, timesteps=args['timesteps'], image_size=args['im_size'], in_channels=in_channels).to(device)

    ckpt = torch.load(os.path.join(model_dir, "best.pth"))
    model.load_state_dict(ckpt["model"])
    # model.eval()

    test_dir = os.path.join(model_dir, "tests")
    os.makedirs(test_dir, exist_ok=True)
    samples = model.sampling(36, clipped_reverse_diffusion=False, device=device, no_noise=False)
    save_image(samples, os.path.join(test_dir, "samples.jpg"), nrow=int(math.sqrt(len(samples))), pad_value=1, normalize=True)
    samples = model.sampling(36, clipped_reverse_diffusion=False, device=device, no_noise=True)
    save_image(samples, os.path.join(test_dir, "deterministic_samples.jpg"), nrow=int(math.sqrt(len(samples))), pad_value=1, normalize=True)

