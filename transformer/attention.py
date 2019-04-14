import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    """
    Implements the scaled dot-product attention as shown in the paper.

    .. note::

        "An attention function can be described as mapping a query and\
        a set of key-value pairs to an output, where the query, keys, values,\
        and output are all vectors. The output is computed as a weighted sum\
        of the values, where the weight assigned to each value is computed\
        by a compatibility function of the query with the corresponding key."



    The overall operation is:

    .. math::

        Attention(Q,K,V) = softmax(\\frac{Q \\cdot K^T}{\sqrt{d_k}}) \\cdot V

    """

    def __init__(self, attn_dropout=0.1):
        """
        Constructor for the ``ScaledDotProductAttention`` class.

        :param attn_dropout: dropout probability.
        :type attn_dropout: float (default = 0.1)

        """

        # call base constructor
        super(ScaledDotProductAttention, self).__init__()

        # instantiate dropout layer
        self.dropout = nn.Dropout(attn_dropout)

        # instantiate softmax layer, along last dimension
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask=None) -> Tensor:
        """
        Implements the forward pass of the ``ScaledDotProductAttention`` class.

        :param queries: Tensor of shape (batch_size, sequence_length, d_k). Represents the queries Q.

        :param keys: Tensor of shape (batch_size, sequence_length, d_k). Represents the keys K.

        :param values: Tensor of shape (batch_size, sequence_length, d_v). Represents the values V.

        :param mask: If not ``None``, will mask out the values at indexes where ``mask==0`` before the softmax layer.

        .. note::

            Can be used in the Decoder to avoid a position at index ``i`` in the sequence to attend to
            positions at indices > ``i``.
            -> prevent a leftward information flow which would be illegal in the ``Decoder``.

        :return: Results of attention weights applied to the values. Shape should be (batch_size, seq_length, d_v)

        """
        # get dimension d_k
        d_k = queries.shape[-1]

        # compute Q * K^T
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(d_k)

        if mask is not None:
            # mask out values at indices where mask is 0 by setting them to a large negative value
            # Will be normalized to 0 in the softmax layer afterwards
            scores = scores.masked_fill(mask == 0, -1e9)

        # get attn weights
        attention_weights = self.softmax(scores)

        attention_weights = self.dropout(attention_weights)

        # apply the weights on the values
        return torch.matmul(attention_weights, values)


class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention. Allows the model to jointly attend\
    to information from different representation subspaces at different positions.

    The implementation is:

    .. math::

        MultiHead(Q, K, V) = Concat(head_1, \\ldots, head_h) \\cdot W^O

        Where:\ head_i = Attention(Q \\cdot W^Q_i, K \\cdot W^K_i, V \\cdot W^V_i)

        W^Q_i, W^K_i \\in \\mathbb{R}^{d_{model} x d_k}

        W^V_i \\in \\mathbb{R}^{d_{model} x d_v}

    The W matrices are linear projections.

    """

    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout=0.1):
        """
        Constructor for the ``MultiHeadAttention`` class.

        :param n_head: number of heads to use (recommended: 8).

        :param d_model: dimension of the output vectors (should be 512).

        :param d_k: Dimensionality of each key / query (Should correspond to d_model / n_head).

        :param d_v: Dimensionality of each value (Should correspond to d_model / n_head).

        :param dropout: dropout probability. Default: 0.1. Passed to :py:class:`ScaledDotProductAttention`.

        """
        # call base constructor
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_head == 0, "Should always have d_model % n_head = 0."
        assert d_k == d_v, "Should always have d_k == d_v."

        # get parameter values
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # instantiate the linear layers (prior to the ScaledDotProductAttention)
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        # instantiate the attention layer
        self.attention = ScaledDotProductAttention(attn_dropout=dropout)

        # final output linear layer
        self.fc = nn.Linear(n_head * d_v, d_model)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask=None) -> Tensor:
        """
        Implements the forward pass of the ``MultiHeadAttention`` class.

        :param queries: Tensor of shape (batch_size, sequence_length, d_model). Represents the queries Q.

        :param keys: Tensor of shape (batch_size, sequence_length, d_model). Represents the keys K.

        :param values: Tensor of shape (batch_size, sequence_length, d_model). Represents the values V.


        .. note ::

            Q, K, V will be passed into linear layers that reduce their dimensionality from d_model to d_k, d_v resp.

            Yet, for efficiency purposes, we pack Q (resp. K,V) of each attention head as one matrix, of dim d_k * n_head.

            Since, in :py:func:`__init__`, we ensure that d_model % n_head = 0, and that we use d_k = d_v = d_model / n_head,

            the input tensors will have dimension d_model.

        :param mask: If not ``None``, will mask out the values at indexes where ``mask==0`` before the softmax layer.
        Can be used in the Decoder to avoid a position at index ``i`` in the sequence to attend to positions
        at indices > ``i`` -> prevent a leftward information flow which would be illegal in the Decoder.

        :return: Results of attention weights applied to the values
                (linear projection of the concatenation of the output of each head).
                Shape should be (batch_size, seq_length, d_model).

        """

        if mask is not None:
            # Same mask applied to all heads.
            mask = mask.unsqueeze(1)

        n_batches = queries.shape[0]

        # 1) Do all the linear projections in batch from d_model => h x d_k
        queries, keys, values = [l(x).view(n_batches, -1, self.n_head, self.d_k).transpose(1, 2)
                                 for l, x in zip((self.w_qs, self.w_ks, self.w_vs), (queries, keys, values))]

        # 2) Apply attention on all the projected vectors in batch.
        x = self.attention(queries, keys, values, mask=mask)

        # 3) Concat using a view.
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.n_head * self.d_k)

        # final linear layer
        return self.fc(x)


if __name__ == '__main__':
    batch_size = 64
    sequence_length = 10
    d_k = d_v = 512

    queries = torch.ones((batch_size, sequence_length, d_k))
    keys = torch.ones((batch_size, sequence_length, d_k))

    values = torch.ones((batch_size, sequence_length, d_v))

    multi_attention = MultiHeadAttention(n_head=8, d_model=512, d_k=64, d_v=64)

    output = multi_attention(queries=queries, keys=keys, values=values)
    print(output.shape)
