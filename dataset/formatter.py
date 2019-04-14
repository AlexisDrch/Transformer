from typing import Optional

import torch
import numpy as np
from torch import Tensor
from torchtext.data import Batch, Dataset, Field

from dataset.europarl import Europarl, Split
from dataset.language_pairs import LanguagePair
from transformer.utils import subsequent_mask

# if CUDA available, moves computations to GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class BatchMasker(Batch):
    """
    Handles the masking in a given batch.
    """

    def __init__(self, batch: Batch, padding_token: str = "<blank>"):
        """
        Constructs a batch masker.

        Takes in a batch, which must have two fields: 'src' and 'trg', and creates the
        source and target masks respectively.

        The created :py:class:`BatchMasker` will thus have the following attributes:

            - `src`: the source sequences (e.g. tokenized input sentences),
            - `trg`: the target sequences (e.g. tokenized output sentences),
            - `src_mask`: Mask hiding the padding in `src`
            - `trg_mask`: Mask hiding both the padding and the subsequent positions in `trg`,
            - `trg_shifted`: Shifted-by-1 targets.

        :param batch: The batch to mask out.
        :param padding_token: The token used to pad shorter sequences.
        """
        super().__init__()
        self.batch = batch

        self.src_field = self.dataset.fields['src']  # type: Field
        self.trg_field = self.dataset.fields['trg']  # type: Field

        # Find integer value of "padding" in the respective vocabularies
        src_padding = self.src_field.vocab.stoi[padding_token]
        trg_padding = self.trg_field.vocab.stoi[padding_token]

        # save source and mask for use during training
        self.src_mask = (self.src != src_padding).to(device)  # type: Tensor
        # Adds a dimension in the middle (equivalent to vec = vec[:,None,:])
        self.src_mask.unsqueeze_(-2)

        self.trg = None  # type: Optional[Tensor]
        self.trg_mask = None  # type: Optional[Tensor]
        self.trg_shifted = None  # type: Optional[Tensor]

        if self.batch.trg is not None:
            self.trg = self.batch.trg[:, :-1].to(device)
            self.trg_shifted = self.batch.trg[:, 1:].to(device)  # type: Tensor

            # create mask to hide padding AND future words (subsequent)
            self.trg_mask = self.make_std_mask(self.trg, trg_padding)

            # ntokens is the size of the sentence (excluding padding)
            self.ntokens = (self.trg_shifted != trg_padding).data.sum()

    @property
    def batch_size(self):
        return self.batch.batch_size

    @property
    def src(self) -> Tensor:
        return self.batch.src

    @property
    def dataset(self) -> Dataset:
        return self.batch.dataset

    @property
    def fields(self):
        return self.batch.fields

    @property
    def input_fields(self):
        return self.batch.input_fields

    @property
    def target_fields(self):
        return self.batch.target_fields

    @staticmethod
    def make_std_mask(target: Tensor, pad) -> Tensor:
        """
        Create a mask for `target` hiding both the padding (specified by `pad`) and the subsequent words
        (prevent token at position i to attend to positions > i).

        :param target: Tensor to create a mask for.

        :param pad: token corresponding to padding elements.

        :return: Mask hiding both padding and subsequent elements in target.
        """
        # hide padding
        target_mask = (target != pad).unsqueeze(-2)

        # hide padding and future words
        target_mask = (target_mask & subsequent_mask(target.shape[-1]).type_as(target_mask.data)).to(device)

        return target_mask

    def cuda(self) -> None:
        """
        Moves Tensors to CUDA.
        """

        self.src.cuda()
        self.src_mask.cuda()
        self.trg.cuda()
        self.trg_mask.cuda()
        self.trg_shifted.cuda()


class Formater(Europarl):
    def __init__(self, language: LanguagePair, split: Split, split_size=0.6, pad=0):
        # call base constructor
        super(Formater, self).__init__(language, split, split_size)
        self.pad = pad

    def __getitem__(self, index):
        # call Europarl get item
        source, target = super(Formater, self).__getitem__(index)
        # tokenize the source and target
        return [self.source_tokenizer(source), self.target_tokenizer(target)]

    def collate_fn(self, batch):
        # Get batch-wise max sequence length
        max_seq_length = np.max([max(len(source), len(target)) for [source, target] in batch])

        # Pad to the max sequence length
        # TODO: Pad the entire batch once with np.pad() if size of pad is diff for each seq
        padded = np.array([[np.pad(arr, (0, max_seq_length - len(arr)), mode='constant',
                                   constant_values=self.pad)
                            for arr in pair] for pair in batch])

        source, target = padded[:, 0, :], padded[:, 1, :]
        # TODO: FIXME
        # return BatchWrapper(Tensor(source), Tensor(target), pad=self.pad)
        raise NotImplementedError()

    def __len__(self):
        return len(self.indexes)
