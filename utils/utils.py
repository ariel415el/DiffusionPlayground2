import math
import os
from time import time

import numpy as np
import torch
import wandb
from torchvision.utils import make_grid


#torchvision ema implementation
#https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L159
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


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
        # save_image(samples, "results/steps_{:0>8}_CLIP.png".format(global_steps), nrow=int(math.sqrt(args.n_samples)),
        #            pad_value=1)
        array = make_grid(samples, nrow=int(math.sqrt(len(samples))), pad_value=1)
        images = self.wandb.Image(array, caption=f"Samples at the end of epoch {epoch}")
        wandb.log({"examples": images})