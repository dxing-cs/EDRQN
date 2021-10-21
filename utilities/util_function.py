import os
import torch
import random
import numpy as np


def set_seed(random_seed):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)


def stable_log_softmax(x):
    shift_x = x - torch.max(x, dim=1, keepdim=True)[0]
    log_s = shift_x - torch.log(torch.sum(torch.exp(shift_x), dim=1, keepdim=True))
    return log_s
