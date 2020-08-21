"""Trains a model on a dataset."""

import numpy as np
from chemprop.args import TrainArgs
from chemprop.train import cross_validate
from chemprop.utils import create_logger
np.random.seed(1)
import torch
torch.manual_seed(1)

#import torch
#torch.manual_seed(1)

if __name__ == '__main__':
    args = TrainArgs().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    cross_validate(args, logger)
