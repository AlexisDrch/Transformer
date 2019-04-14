import torch
import numpy as np
from torch.utils.data import Dataset

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class CopyTaskDataset(Dataset):
    """
    A simple copy task: The network is asked to produce outputs similar to the inputs.

    This is used as a basic test to ensure gradients are flowing correctly through the network, and the latter
    is able to overfit a small amount of data.
    """
    def __init__(self, max_int: int, max_seq_length: int, size: int,):
        """
        Constructor of the ``CopyTaskDataset``.

        :param max_int: Upper bound on the randomly drawn samples.

        :param max_seq_length: Sequence length. Will be the same for all samples.

        :param size: Size of the dataset. Mainly used for ``__len__`` as generated samples are random (and bounded by ``max_len``).
        """
        self.max_int = max_int
        self.max_seq_length = max_seq_length
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        """
        Randomly creates a sample of shape [1, self.max_seq_length], where the elements are drawn randomly
        in U[0, self.max_int].

        As this is a copy task, inputs = targets.

        :param item: index of the sample, not used here.

        :return: tuple (inputs, targets) of identical shape.
        """
        return {}

    def collate(self, samples):

        data = torch.from_numpy(np.random.randint(1, self.max_int, size=(len(samples), self.max_seq_length)))
        data[:, 0] = 1

        return Batch(data, data, 0)


class Batch:
    """
    Small class which represents a batch, holding the inputs & outputs sequences, and creates the associated batch.
    """

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg != pad).unsqueeze(-2)  # will be ANDed with subsequent_mask(trg.shape[1]) in the forward.

    def cuda(self) -> None:
        """
        CUDAify the Batch parameters which can be CUDAified (i.e. tensors).

        """
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()
        self.trg = self.trg.cuda()
        self.trg_y = self.trg_y.cuda()
        self.trg_mask = self.trg_mask.cuda()
