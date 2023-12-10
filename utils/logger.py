import math
import os
from time import time

import numpy as np
import wandb
from torchvision.utils import save_image, make_grid


class Logger:
    def __init__(self, args):
        self.args = args
        self.out_dir = os.path.join("results", args.train_name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.wandb = wandb.init(project='DDPM', dir=self.out_dir, name=args.train_name)
        self.best_loss = np.inf
        self.start = time()

    def log(self, loss, epoch, iter, lr):
        self.wandb.log({"Loss": loss})
        if iter % self.args.log_freq == 0:
            start_iteration = 0 # TODO load from ckpt
            it_sec = max(1, iter - start_iteration) / (time() - self.start)
            print(f"Epoch[{epoch}/{self.args.epochs}],Step[{iter}],loss:{loss:.5f},lr:{lr:.5f} it/sec: {it_sec:.1f}")

    def is_best_loss(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            return True
        return False

    def plot(self, samples, epoch):
        save_image(samples, os.path.join(self.out_dir, f"{epoch}.jpg"), nrow=int(math.sqrt(len(samples))), pad_value=1)
        array = make_grid(samples, nrow=int(math.sqrt(len(samples))), pad_value=1)
        images = wandb.Image(array, caption=f"Samples at the end of epoch {epoch}")
        wandb.log({"examples": images})
