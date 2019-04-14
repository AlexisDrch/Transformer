import torch
from torch import nn, Tensor, tensor

# if CUDA available, moves computations to GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    FloatTensor = torch.cuda.FloatTensor
else:
    device = torch.device('cpu')
    FloatTensor = torch.FloatTensor


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        Creates a word embeddings.

        Note that for the transformer model, the input embeddings
        is constrained to using the same weight matrix as the output transformation.

        :param d_model: The dimension of the output to use.
        :param vocab_size: The size of the vocabulary.
        """
        super(Embeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.d_model_sqrt = torch.sqrt(tensor([d_model]).type(FloatTensor))

    def forward(self, x):
        """
        Forward pass of the :py:class:`Embeddings` class.

        Returns the following:

        .. math::

            self.look\_up\_table(x) \\cdot \sqrt{d\_model}

        :param x: Input tensor to embed. Shape is arbitrary, but should be close to [batch_size, sequence_length].

        :return: Embedded x, of shape [x.shape, d_model].
        """

        return self.embeddings(x).type(FloatTensor) * self.d_model_sqrt


class PositionalEncoding(nn.Module):
    """
    As the model contains no recurrence & convolution, it uses positional encodings\
    to make use of the order of the sequence.

    These positional encodings represent information about the relative or absolute position
    of the tokens in the sequence. They have the same dimension d_model as the ``Embeddings`` so that the 2 are summed.

    The equations are:

    .. math::

        PE(pos,2 \\cdot i) = sin(\\frac{pos}{10000^{\\frac{2 \\cdot i}{d_{model}}}})


        PE(pos,2 \\cdot i + 1) = cos(\\frac{pos}{10000^{\\frac{2 \\cdot i}{d_{model}}}})



    where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds
    to a sinusoid.b The wavelengths form a geometric progression from 2π to 10000⋅2π.

    In addition, dropout is applied to the sums of the embeddings and the positional encodings in both the
    ``Encoder`` and ``Decoder``.

    """

    def __init__(self, d_model: int, dropout: float, max_len=5000):
        """
        Instantiate the :py:class:`PositionalEncoding`.

        :param d_model: Overall model dimension (Should be 512).

        :param dropout: Dropout probability (the paper mentions 0.1).
                Dropout is applied after the sum with the positional encodings.

        :param max_len: Maximum sequence length.

        """
        # call base constructor.
        super(PositionalEncoding, self).__init__()

        # dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pos_encoding = torch.zeros(max_len, d_model, device=device)

        position = torch.arange(0., max_len, device=device).unsqueeze(1)  # shape will be (max_len, 1)

        # division term: use exponential & log for numerical stability?
        div_term = torch.exp(torch.arange(0., d_model, 2, device=device) * -(torch.log(tensor([10000.0], device=device)) / d_model))

        # even dimension: sinusoid
        pos_encoding[:, 0::2] = torch.sin(position * div_term)

        # odd dimension: cosinusoid
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        pos_encoding = pos_encoding.unsqueeze(0)
        # final shape will be (1, max_length, d_model)

        # register the positional encoding into the persistent state of the model.
        # They are not considered a model parameter, but can be accessed as an attribute with their name.
        self.register_buffer(name='pos_encoding', tensor=pos_encoding)

        # indicate positional encoding are not learnable, thus do not need gradients.
        self.pos_encoding.requires_grad = False

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Forward pass of the `:py:class:`PositionalEncoding`.

        :param embeddings: Input tensor, representing the current word embeddings. Should be of shape (batch_size, seq_len, d_model).

        :return: Sum of embeddings and positional encodings (with dropout applied). Should be the same shape as `embeddings`.
        """
        # add positional encoding (up to the sequence length) to embeddings
        return self.dropout(embeddings + self.pos_encoding[:, :embeddings.shape[1]])
