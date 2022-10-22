import os
import random

import numpy as np
import torch


def seed_everything(args):
    random.seed(args)
    np.random.seed(args)
    os.environ["PYTHONHASHSEED"] = str(args)
    torch.manual_seed(args)
    torch.cuda.manual_seed(args)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True