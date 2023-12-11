import json
import math
import os
from time import time

import numpy as np
import wandb
from torchvision.utils import save_image, make_grid


class Logger:
    def __init__(self, args):
        self.args = args
        self.out_dir = os.path.join("outputs", args.project_name, args.train_name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.wandb = wandb.init(project='DDPM', dir=self.out_dir, name=args.train_name)
        self.best_loss = np.inf
        self.start = time()
        with open(os.path.join(self.out_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def log(self, loss, epoch, global_step, lr):
        self.wandb.log({"Loss": loss})
        if global_step % self.args.print_freq == 0:
            start_iteration = 0 # TODO load from ckpt
            it_sec = max(1, global_step - start_iteration) / (time() - self.start)
            print(f"Global step: {global_step} (Epoch[{epoch}]),loss:{loss:.5f},lr:{lr:.5f} it/sec: {it_sec:.1f}")

    def is_best_loss(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            return True
        return False

    def plot(self, samples, global_step):
        save_image(samples, os.path.join(self.out_dir, f"{global_step}.jpg"), nrow=int(math.sqrt(len(samples))), pad_value=1)
        array = make_grid(samples, nrow=int(math.sqrt(len(samples))), pad_value=1)
        images = wandb.Image(array, caption=f"Samples at the end of Step {global_step}")
        wandb.log({"examples": images})
