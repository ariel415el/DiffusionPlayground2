import torch


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


def parse_classnames_and_kwargs(string, kwargs=None):
    """Import a class and create an instance with kwargs from strings in format
    '<class_name>_<kwarg_1_name>=<kwarg_1_value>_<kwarg_2_name>=<kwarg_2_value>"""
    name_and_args = string.split('-')
    class_name = name_and_args[0]
    if kwargs is None:
        kwargs = dict()
    for arg in name_and_args[1:]:
        name, value = arg.split("=")
        kwargs[name] = value
    return class_name, kwargs
