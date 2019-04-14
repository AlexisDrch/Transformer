import copy

import numpy as np
import torch
import torch.nn as nn

# if CUDA available, moves computations to GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def clone(module, N) -> nn.ModuleList:
    """
    Produces ``N`` identical copies of ``module`` and returns them as a ``nn.ModuleList``.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """
    Masks out subsequent positions.

    The mask shows the position each tgt word (row) is allowed to look at (column).
    Words are blocked for attending to future words during training.

    :param size: Input size
    :return: Tensor with boolean mask on subsequent position
    """
    attn_shape = (1, size, size)
    # pylint: disable=no-member
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangle of an array.

    return torch.from_numpy(subsequent_mask).to(device) == 0


class BColors:
    """
    Pre defined colors for console output
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
