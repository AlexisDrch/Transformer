import logging
from os.path import join
from typing import Optional, Union

import torch
import torch.nn as nn
from datetime import datetime
from transformer.utils import subsequent_mask
from transformer.encoder import Encoder, EncoderLayer
from transformer.decoder import Decoder, DecoderLayer
from transformer.attention import MultiHeadAttention
from transformer.layers import PositionwiseFeedForward
from transformer.classifier import OutputClassifier
from transformer.embeddings import Embeddings, PositionalEncoding

# if CUDA available, change some tensor types to move computations to GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    IntTensor = torch.cuda.IntTensor
else:
    device = torch.device('cpu')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    IntTensor = torch.IntTensor


class Transformer(nn.Module):
    """
    Main class for the Transformer model.
    Please see https://arxiv.org/abs/1706.03762 for the reference paper.
    """

    def __init__(self, params: dict):
        """
        Instantiate the ``Transformer`` class.

        :param params: Dict containing the set of parameters for the entire model\
         (e.g ``EncoderLayer``, ``DecoderLayer`` etc.) broken down in relevant sections, e.g.:

            params = {
                'd_model': 512,
                'src_vocab_size': 27000,
                'tgt_vocab_size': 27000,

                'N': 6,
                'dropout': 0.1,

                'attention': {'n_head': 8,
                              'd_k': 64,
                              'd_v': 64,
                              'dropout': 0.1},

                'feed-forward': {'d_ff': 2048,
                                 'dropout': 0.1},
            }

        """
        # call base constructor
        super(Transformer, self).__init__()

        # Save params for Checkpoint
        self._params = params

        # instantiate Encoder layer
        enc_layer = EncoderLayer(size=params['d_model'],
                                 self_attention=MultiHeadAttention(n_head=params['attention']['n_head'],
                                                                   d_model=params['d_model'],
                                                                   d_k=params['attention']['d_k'],
                                                                   d_v=params['attention']['d_v'],
                                                                   dropout=params['attention']['dropout']),
                                 feed_forward=PositionwiseFeedForward(d_model=params['d_model'],
                                                                      d_ff=params['feed-forward']['d_ff'],
                                                                      dropout=params['feed-forward']['dropout']),
                                 dropout=params['dropout'])

        # instantiate Encoder
        self.encoder = Encoder(layer=enc_layer, n_layers=params['N'])

        # instantiate Decoder layer
        decoder_layer = DecoderLayer(size=params['d_model'],
                                     self_attn=MultiHeadAttention(n_head=params['attention']['n_head'],
                                                                   d_model=params['d_model'],
                                                                   d_k=params['attention']['d_k'],
                                                                   d_v=params['attention']['d_v'],
                                                                   dropout=params['attention']['dropout']),
                                     memory_attn=MultiHeadAttention(n_head=params['attention']['n_head'],
                                                                   d_model=params['d_model'],
                                                                   d_k=params['attention']['d_k'],
                                                                   d_v=params['attention']['d_v'],
                                                                   dropout=params['attention']['dropout']),
                                     feed_forward=PositionwiseFeedForward(d_model=params['d_model'],
                                                                      d_ff=params['feed-forward']['d_ff'],
                                                                      dropout=params['feed-forward']['dropout']),
                                     dropout=params['dropout'])

        # instantiate Decoder
        self.decoder = Decoder(layer=decoder_layer, N=params['N'])

        pos_encoding = PositionalEncoding(d_model=params['d_model'], dropout=params['dropout'])

        self.src_embeddings = nn.Sequential(Embeddings(d_model=params['d_model'], vocab_size=params['src_vocab_size']),
                                            pos_encoding)

        self.trg_embeddings = nn.Sequential(Embeddings(d_model=params['d_model'], vocab_size=params['tgt_vocab_size']),
                                            pos_encoding)

        self.classifier = OutputClassifier(d_model=params['d_model'], vocab=params['tgt_vocab_size'])

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_sequences, src_mask, trg_sequences, trg_mask) -> torch.Tensor:
        """
        Main forward pass of the model. Simplified worfklow:

        src_sequences -> Embeddings + Positional Encodings -> Encoder stack -> Memory --|
                                                                 |----------------------|
                                                                 v
        trg_sequences -> Embeddings + Positional Encodings -> Decoder stack -> Classifier -> Logits


        :param src_sequences: Batch of tokenized input sentences. Should be of shape (batch_size, in_seq_len).

        :param  src_mask: Mask, hiding the padding in the input batch. Should be same shape as src_sequences.

        :param trg_sequences: Batch of output sentences. Should be of shape (batch_size, out_seq_len).

        :param trg_mask: Mask, hiding the padding in the output batch. Should be the same shape as trg_sequences.

        .. note::

            This mask (which hides padding) will be combined with the `subsequent_mask` which hides subsequent
            positions in the decoder, to form only one mask.


        :return: Logits, of shape (batch_size, out_seq_len, d_model)
        """

        # 1. embed the input batch: have to move input sequences to torch.*.LongTensor
        src_sequences = self.src_embeddings(src_sequences.type(LongTensor))

        # 2. encoder stack
        encoder_output = self.encoder(src=src_sequences, mask=src_mask, verbose=False)

        # 3. get subsequent mask to hide subsequent positions in the decoder.
        # self_mask = subsequent_mask(trg_sequences.shape[1])

        # 3.5 Combine the trg_mask (which hides padding) and self_mask (which hide subsequent positions in the decoder)
        # as one mask

        # hide_padding_and_future_words_mask = trg_mask.type_as(self_mask.data) & self_mask

        # 4. embed the output batch
        trg_sequences = self.trg_embeddings(trg_sequences.type(LongTensor))

        # 4. decoder stack
        decoder_output = self.decoder(x=trg_sequences, memory=encoder_output,
                                      self_mask=trg_mask, memory_mask=src_mask)

        # 5. classifier
        logits = self.classifier(decoder_output)

        return logits

    def greedy_decode(self, src: torch.Tensor, src_mask: torch.Tensor, trg_vocab, start_symbol="<s>", stop_symbol="</s>", max_length=100) -> torch.Tensor:
        """
        Returns the prediction for `src` using greedy decoding for simplicity:

            - Feed `src` (after embedding) in the Encoder to get the "memory",
            - Feed an initial tensor (filled with start_symbol) in the Decoder, with the "memory" and the appropriate corresponding mask
            - Get the predictions of the model, makes a max to get the next token, cat it to the previous prediction and iterate


        :param src: sample for which to produce predictions.

        :param src_mask: Associated `src` mask

        :param trg_vocab: Vocabulary set of the target sentences.
        :type trg_vocab: torchtext.vocab.Vocab

        :param start_symbol: Symbol used as initial value for the Decoder. Should correspond to start_token="<s>" in the dataset vocab).

        :param stop_symbol: Symbol used to represent an end of sentence, e.g. "</s>" (in the dataset vocab).

        :param max_length: Maximum sequence length of the prediction.

        """
        # 0. Ensure inference mode
        self.eval()

        # 1. Embed src
        embedded = self.src_embeddings(src.type(LongTensor))

        # 2. Encode embedded inputs
        memory = self.encoder(src=embedded, mask=src_mask)

        # 3. Create initial input for decoder
        decoder_in = torch.ones(src.shape[0], 1).type(FloatTensor) * trg_vocab.stoi[start_symbol]

        for i in range(max_length):

            # 4. Embed decoder_in
            decoder_in_embed = self.trg_embeddings(decoder_in.type(LongTensor))

            # 5. Go through decoder
            out = self.decoder(x=decoder_in_embed, memory=memory,
                               self_mask=subsequent_mask(decoder_in.shape[1]),
                               memory_mask=src_mask)

            # 6. classifier: TODO: Why only last word?
            logits = self.classifier(out[:, -1])

            # 7. Get predicted token for each sample in the batch
            _, next_token = logits.max(dim=1, keepdim=True)

            # 8. Concatenate predicted token with previous predictions
            decoder_in = torch.cat([decoder_in, next_token.type(FloatTensor)], dim=1)

        # cast to int tensors
        decoder_in = decoder_in.type(IntTensor)
        # 9. retrieve words from tokens in the target vocab
        translation = ""
        for i in range( decoder_in.shape[1]):
            sym = trg_vocab.itos[decoder_in[0, i]]
            translation += sym + " "

            if sym == trg_vocab.stoi[stop_symbol]:
                break

        # 10. return prediction
        return translation

    def save(self, model_dir: str, epoch_idx: int, loss_value: float, model_name: str = None) -> str:
        """
        Method to save a model along with a couple of information: number of training epochs and reached loss.

        # TODO: Could be extended if wish to save more statistics and state of model (e.g. 'converged' or not).

        :param model_dir: Directory where the model will be saved.

        :param epoch_idx: Epoch number.

        :param loss_value: Reached loss value at end of epoch ``epoch_idx``.

        :returns: The path to the file
        """

        # Checkpoint to be saved.
        chkpt = {
            'name': 'Transformer',
            'params': self._params,
            'state_dict': self.state_dict(),
            'model_timestamp': datetime.now(),
            'epoch': epoch_idx,
            'loss': loss_value,
        }

        if model_name is None:
            model_name = f"model_epoch_{epoch_idx}.pt"

        filename = join(model_dir, model_name)
        torch.save(chkpt, filename)
        return filename

    def load(self, checkpoint: Union[str, dict], logger: Optional[logging.Logger] = None) -> None:
        """
        Loads a model from the specified checkpoint file.

        :param checkpoint: Dictionary with model state and statistics,
                           or a path to a checkpoint file.

        :param: logger: Logger object (to indicate number of trained epochs and loss value from loaded model).

        """
        # Load checkpoint
        # This is to be able to load a CUDA-trained model on CPU
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        assert isinstance(checkpoint, dict), ("The checkpoint must be a dictionary or at least "
                                              "a path to a checkpoint file.")

        # Load model.
        self.load_state_dict(checkpoint['state_dict'])

        # Print statistics.
        if logger is not None:
            logger.info(
                "Imported Transformer parameters from checkpoint from {} (epoch: {}, loss: {})".format(
                    checkpoint['model_timestamp'],
                    checkpoint['epoch'],
                    checkpoint['loss'],
                    ))

    @staticmethod
    def load_model_from_file(checkpoint_file, logger: Optional[logging.Logger] = None,
                             params: Optional[dict] = None):
        """
        This method is similar to the `load` method, but is static and as such does not need
        an already instantiated model.

        :param checkpoint_file: The path to a checkpoint file.
        :param logger: An optional logger to log the values in the checkpoint.
        :param params: If not None, those are used in place of the params in the checkpoint.

        :return: A Transformer model.
        :rtype: Transformer
        """
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        if params is None:
            if not 'params' in checkpoint:
                raise ValueError("The checkpoint does not contain the model params. "
                                 "It might have be saved with an older version of the code.\n"
                                 "Please instantiate a Transformer first and use "
                                 "the `load` instance method on it instead.")
            params = checkpoint['params']
        model = Transformer(params=params)
        model.load(checkpoint, logger)
        return model