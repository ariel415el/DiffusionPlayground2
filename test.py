import argparse
import json
import math
import os

import torch
from torchvision.utils import save_image

from DDPM import DDPM


if __name__ == '__main__':
    device = torch.device('cuda:0')
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('train_name', type=str)
    args = parser.parse_args()

    model_dir = os.path.join("results", args.train_name)
    args = json.load(open(os.path.join(model_dir, "args.txt")))

    model = DDPM(timesteps=args['timesteps'],
                           image_size=args['im_size'],
                           in_channels=1 if args['gray_scale'] else 3,
                           base_dim=args['model_base_dim'],
                           dim_mults=[2, 4, 8, 16]).to(device)

    ckpt = torch.load(os.path.join(model_dir, "best.pth"))
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_dir = os.path.join(model_dir, "tests")
    os.makedirs(test_dir, exist_ok=True)
    samples = model.sampling(16, clipped_reverse_diffusion=False, device=device)
    save_image(samples, os.path.join(test_dir, "samples.jpg"), nrow=int(math.sqrt(len(samples))), pad_value=1)

