import torch.nn as nn
from torch import Tensor


class OutputClassifier(nn.Module):
    """
    1-layer feed forward network used to make next-token probability predictions on the output vocabulary.
    """
    def __init__(self, d_model, vocab):
        """
        Implement the ``OutputClassifier`` class.

        :param d_model: size of the input vectors (Should be the overall model dimension = 512).

        :param vocab: size of the output vocabulary set.
        """
        # call base constructor
        super(OutputClassifier, self).__init__()

        # only one linear layer
        self.linear1 = nn.Linear(d_model, vocab)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ``OutputClassifier``.

        Simply goes through a linear layer and return logits (i.e. raw scores, not normalized probabilities).

        :param x: Input Tensor, which should come from the ``Decoder``. Shape should be (batch_size, seq_len, d_model).

        :return: Next token raw scores, of shape (batch_size, seq_len, vocab).
        """
        return self.linear1(x)
