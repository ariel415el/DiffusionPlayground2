import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from DDPM import DDPM
from models import Unet, FC
from utils.utils import ExponentialMovingAverage
import os
import argparse

from utils.data import create_mnist_dataloaders
from utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--project_name', type=str, default='DDPM')
    parser.add_argument('--train_name', type=str, default='test_DDPM')

    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--limit_data', default=None, type=int)
    parser.add_argument('--center_crop', default=None, help='center_crop_data to specified size', type=int)
    parser.add_argument('--gray_scale', action='store_true', default=False, help="Load data as grayscale")
    parser.add_argument('--im_size', type=int, default=64)

    parser.add_argument('--arch', type=str, default='unet')
    parser.add_argument('--denoiser_depth', type=int, default=2)
    parser.add_argument('--model_base_dim', type=int, help='base dim of Unet', default=64)
    parser.add_argument('--timesteps', type=int, help='sampling steps of DDPM', default=1000)
    parser.add_argument('--model_ema_steps', type=int, help='ema model evaluation interval', default=10)
    parser.add_argument('--model_ema_decay', type=float, help='ema model decay', default=0.995)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10000)

    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--ckpt', type=str, help='define checkpoint path', default='')
    parser.add_argument('--n_samples', type=int, help='define sampling amounts after every epoch trained', default=36)
    parser.add_argument('--print_freq', type=int, help='training log message printing frequence', default=10)
    parser.add_argument('--log_freq', type=int, help='training log message printing frequence', default=5000)
    parser.add_argument('--cpu', action='store_true', help='cpu training')

    args = parser.parse_args()
    print("Command line arguments:")
    for k,v in vars(args).items():
        print(f"\t- {k}: {v}")
    return args


def get_model(args, device):
    in_channels = 1 if args.gray_scale else 3
    if args.arch == 'unet':
        denoiser = Unet(args.timesteps, 256, in_channels, in_channels, args.model_base_dim, depth=args.denoiser_depth)
    elif args.arch == "fc":
        denoiser = FC(args.image_size, args.timesteps, 256, in_channels, in_channels, args.model_base_dim, args.denoiser_depth)
    else:
        ValueError("bad architecture name")
    model = DDPM(denoiser, timesteps=args.timesteps,image_size=args.im_size, in_channels=in_channels).to(device)

    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    return model, model_ema


def main(args):
    logger = Logger(args)
    device = "cpu" if args.cpu else "cuda"
    train_dataloader = create_mnist_dataloaders(args)

    model, model_ema = get_model(args, device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(optimizer, args.lr, total_steps=args.epochs * len(train_dataloader), pct_start=0.25,
                           anneal_strategy='cos')
    loss_fn = nn.MSELoss(reduction='mean')

    # load checkpoint


    global_steps = 0
    for epoch in range(args.epochs):
        model.train()
        for image in train_dataloader:
            noise = torch.randn_like(image).to(device)
            image = image.to(device)
            pred = model(image, noise)
            loss = loss_fn(pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if global_steps % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
            global_steps += 1
            loss = loss.detach().cpu().item()
            logger.log(loss, epoch, global_steps, scheduler.get_last_lr()[0])

            if global_steps % args.log_freq == 0:
                if logger.is_best_loss(loss):
                    ckpt = {"model": model.state_dict(),
                        "model_ema": model_ema.state_dict()}
                    torch.save(ckpt, os.path.join(logger.out_dir, "best.pth"))

                model_ema.eval()
                samples = model_ema.module.sampling(args.n_samples, clipped_reverse_diffusion=True, device=device)
                logger.plot(samples, global_steps)


if __name__ == "__main__":
    args = parse_args()
    main(args)
