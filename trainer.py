import json
import logging
import logging.config
import os
import random
from datetime import datetime
from os.path import join

import torch
from google.cloud import storage

from dataset.iwslt import IWSLTDatasetBuilder
from dataset.language_pairs import LanguagePair
from dataset.utils import Split
from training.loss import LabelSmoothingLoss, CrossEntropyLoss
from training.optimizer import NoamOpt
from training.statistics_collector import StatisticsCollector
from transformer.model import Transformer

HYPERTUNER = None


class Trainer(object):
    """
    Represents a worker taking care of the training of an instance of the ``Transformer`` model.

    """

    def __init__(self, params: dict):
        """
        Constructor of the Trainer.
        Sets up the following:
            - Device available (e.g. if CUDA is present)
            - Initialize the model, dataset, loss, optimizer
            - log statistics (epoch, elapsed time, BLEU score etc.)
        """
        self.is_hyperparameter_tuning = HYPERTUNER is not None
        self.gcs_job_dir = params["settings"].get("save_dir", None)
        self.model_name = params["settings"].get("model_name", None)

        # configure all logging
        self.configure_logging(training_problem_name="IWSLT")

        # set all seeds
        self.set_random_seeds(pytorch_seed=params["settings"]["pytorch_seed"],
                              numpy_seed=params["settings"]["numpy_seed"],
                              random_seed=params["settings"]["random_seed"])

        # Initialize TensorBoard and statistics collection.
        self.initialize_statistics_collection()

        if self.is_hyperparameter_tuning:
            assert "save_dir" in params["settings"] and "model_name" in params["settings"], \
                "Expected parameters 'save_dir' and 'model_name'."
        else:
            # save the configuration as a json file in the experiments dir
            with open(self.log_dir + 'params.json', 'w') as fp:
                json.dump(params, fp)
            self.logger.info('Configuration saved to {}.'.format(self.log_dir + 'params.json'))

        self.initialize_tensorboard(log_dir=self.gcs_job_dir)

        # initialize training Dataset class
        self.logger.info("Creating the training & validation dataset, may take some time...")
        (self.training_dataset_iterator, self.validation_dataset_iterator,
         self.test_dataset_iterator, self.src_vocab, self.trg_vocab) = (
            IWSLTDatasetBuilder.build(
                language_pair=LanguagePair.fr_en,
                split=Split.Train | Split.Validation | Split.Test,
                max_length=params["dataset"]["max_seq_length"],
                min_freq=params["dataset"]["min_freq"],
                start_token=params["dataset"]["start_token"],
                eos_token=params["dataset"]["eos_token"],
                blank_token=params["dataset"]["pad_token"],
                batch_size_train=params["training"]["train_batch_size"],
                batch_size_validation=params["training"]["valid_batch_size"],
            )
        )

        # get the size of the vocab sets
        self.src_vocab_size, self.trg_vocab_size = len(self.src_vocab), len(self.trg_vocab)

        # Find integer value of "padding" in the respective vocabularies
        self.src_padding = self.src_vocab.stoi[params["dataset"]["pad_token"]]
        self.trg_padding = self.trg_vocab.stoi[params["dataset"]["pad_token"]]

        # just for safety, assume that the padding token of the source vocab is always equal to the target one (for now)
        assert self.src_padding == self.trg_padding, (
            "the padding token ({}) for the source vocab is not equal "
            "to the one from the target vocab ({})."
                .format(self.src_padding, self.trg_padding)
        )

        self.logger.info(
            "Created a training & a validation dataset, with src_vocab_size={} and trg_vocab_size={}"
                .format(self.src_vocab_size, self.trg_vocab_size))

        # pass the size of input & output vocabs to model's params
        params["model"]["src_vocab_size"] = self.src_vocab_size
        params["model"]["tgt_vocab_size"] = self.trg_vocab_size

        # can now instantiate model
        self.model = Transformer(params["model"])  # type: Transformer

        if params["training"].get("multi_gpu", False):
            self.model = torch.nn.DataParallel(self.model)
            self.logger.info(
                "Multi-GPU training activated, on devices: {}".format(self.model.device_ids))
            self.multi_gpu = True
        else:
            self.multi_gpu = False

        if params["training"].get("load_trained_model", False):
            if self.multi_gpu:
                self.model.module.load(
                    checkpoint=params["training"]["trained_model_checkpoint"],
                    logger=self.logger)
            else:
                self.model.load(checkpoint=params["training"]["trained_model_checkpoint"],
                                logger=self.logger)

        if torch.cuda.is_available():
            self.model = self.model.cuda()  # type: Transformer

        # whether to save the model at every epoch or not
        self.save_intermediate = params["training"].get("save_intermediate", False)

        # instantiate loss
        if "smoothing" in params["training"]:
            self.loss_fn = LabelSmoothingLoss(size=self.trg_vocab_size,
                                              padding_token=self.src_padding,
                                              smoothing=params["training"]["smoothing"])
            self.logger.info(f"Using LabelSmoothingLoss with "
                             f"smoothing={params['training']['smoothing']}.")
        else:
            self.loss_fn = CrossEntropyLoss(pad_token=self.src_padding)
            self.logger.info("Using CrossEntropyLoss.")

        # instantiate optimizer
        self.optimizer = NoamOpt(model=self.model,
                                 model_size=params["model"]["d_model"],
                                 lr=params["optim"]["lr"],
                                 betas=params["optim"]["betas"],
                                 eps=params["optim"]["eps"],
                                 factor=params["optim"]["factor"],
                                 warmup=params["optim"]["warmup"],
                                 step=params["optim"]["step"])

        # get number of epochs and related hyper parameters
        self.epochs = params["training"]["epochs"]

        self.logger.info('Experiment setup done.')

    def train(self):
        """
        Main training loop.

            - Trains the Transformer model on the specified dataset for a given number of epochs
            - Logs statistics to logger for every batch per epoch

        """
        # Reset the counter.
        episode = -1
        val_loss = 0.

        for epoch in range(self.epochs):

            # Empty the statistics collectors.
            self.training_stat_col.empty()
            self.validation_stat_col.empty()

            # collect epoch index
            self.training_stat_col['epoch'] = epoch + 1
            self.validation_stat_col['epoch'] = epoch + 1

            # ensure train mode for the model
            self.model.train()

            for i, batch in enumerate(
                IWSLTDatasetBuilder.masked(
                    IWSLTDatasetBuilder.transposed(
                        self.training_dataset_iterator
                    ))):

                # "Move on" to the next episode.
                episode += 1

                # 1. reset all gradients
                self.optimizer.zero_grad()

                # Convert batch to CUDA.
                if torch.cuda.is_available():
                    batch.cuda()

                # 2. Perform forward pass.
                logits = self.model(batch.src, batch.src_mask, batch.trg, batch.trg_mask)

                # 3. Evaluate loss function.
                loss = self.loss_fn(logits, batch.trg_shifted)

                # 4. Backward gradient flow.
                loss.backward()

                # 4.1. Export to csv - at every step.
                # collect loss, episode
                self.training_stat_col['loss'] = loss.item()
                self.training_stat_col['episode'] = episode
                self.training_stat_col['src_seq_length'] = batch.src.shape[1]
                self.training_stat_col.export_to_csv()

                # 4.2. Exports statistics to the logger.
                self.logger.info(self.training_stat_col.export_to_string())

                # 4.3 Exports to tensorboard
                self.training_stat_col.export_to_tensorboard()

                # 5. Perform optimization step.
                self.optimizer.step()

            # save model at end of each epoch if indicated:
            if self.save_intermediate:
                if self.multi_gpu:
                    self.model.module.save(self.model_dir, epoch, loss.item())
                else:
                    self.model.save(self.model_dir, epoch, loss.item())
                self.logger.info("Model exported to checkpoint.")

            # validate the model on the validation set
            self.model.eval()
            val_loss = 0.

            with torch.no_grad():
                for i, batch in enumerate(
                    IWSLTDatasetBuilder.masked(
                        IWSLTDatasetBuilder.transposed(
                            self.validation_dataset_iterator
                        ))):

                    # Convert batch to CUDA.
                    if torch.cuda.is_available():
                        batch.cuda()

                    # 1. Perform forward pass.
                    logits = self.model(batch.src, batch.src_mask, batch.trg, batch.trg_mask)

                    # 2. Evaluate loss function.
                    loss = self.loss_fn(logits, batch.trg_shifted)

                    # Accumulate loss
                    val_loss += loss.item()

            # 3.1 Collect loss, episode: Log only one point per validation (for now)
            self.validation_stat_col['loss'] = val_loss / (i + 1)
            self.validation_stat_col['episode'] = episode

            # 3.1. Export to csv.
            self.validation_stat_col.export_to_csv()

            # 3.2 Exports statistics to the logger.
            self.logger.info(self.validation_stat_col.export_to_string('[Validation]'))

            # 3.3 Export to Tensorboard
            self.validation_stat_col.export_to_tensorboard()

            # 3.4 Save model on GCloud
            if self.gcs_job_dir is not None:
                if self.multi_gpu:
                    filename = self.model.module.save(self.model_dir, epoch, loss.item(),
                                                      model_name=self.model_name)
                else:
                    filename = self.model.save(self.model_dir, epoch, loss.item(),
                                               model_name=self.model_name)
                save_model(self.gcs_job_dir, filename, self.model_name)

            # 3.4.b Export to Hypertune
            if self.is_hyperparameter_tuning:
                assert HYPERTUNER is not None
                HYPERTUNER.report_hyperparameter_tuning_metric(
                    hyperparameter_metric_tag='validation_loss',
                    metric_value=val_loss / (i + 1),
                    global_step=epoch)

        # always save the model at end of training
        if self.multi_gpu:
            self.model.module.save(self.model_dir, epoch, loss.item())
        else:
            self.model.save(self.model_dir, epoch, loss.item())

        self.logger.info("Final model exported to checkpoint.")

        # training done, end statistics collection
        self.finalize_statistics_collection()
        self.finalize_tensorboard()

        return val_loss

    def configure_logging(self, training_problem_name: str, logger_config=None) -> None:
        """
        Takes care of the initialization of logging-related objects:

            - Sets up a logger with a specific configuration,
            - Sets up a logging directory
            - sets up a logging file in the log directory
            - Sets up a folder to store trained models

        :param training_problem_name: Name of the dataset / training task (e.g. "copy task", "IWLST"). Used for the logging
        folder name.
        """
        # instantiate logger
        # Load the default logger configuration.
        if logger_config is None:
            logger_config = {'version': 1,
                             'disable_existing_loggers': False,
                             'formatters': {
                                 'simple': {
                                     'format': '[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
                                     'datefmt': '%Y-%m-%d %H:%M:%S'}},
                             'handlers': {
                                 'console': {
                                     'class': 'logging.StreamHandler',
                                     'level': 'INFO',
                                     'formatter': 'simple',
                                     'stream': 'ext://sys.stdout'}},
                             'root': {'level': 'DEBUG',
                                      'handlers': ['console']}}
            if self.is_hyperparameter_tuning or self.gcs_job_dir is not None:
                # Running on GCloud, use shorter messages as time and debug level will be
                # saved elsewhere.
                logger_config['formatters']['simple']['format'] = "%(name)s >>> %(message)s"

        logging.config.dictConfig(logger_config)

        # Create the Logger, set its label and logging level.
        self.logger = logging.getLogger(name='Trainer')

        # Prepare the output path for logging
        time_str = '{0:%Y%m%d_%H%M%S}'.format(datetime.now())
        self.log_dir = 'experiments/' + training_problem_name + '/' + time_str + '/'

        os.makedirs(self.log_dir, exist_ok=False)
        self.logger.info('Folder {} created.'.format(self.log_dir))

        # Set log dir and add the handler for the logfile to the logger.
        self.log_file = self.log_dir + 'training.log'
        self.add_file_handler_to_logger(self.log_file)

        self.logger.info('Log File {} created.'.format(self.log_file))

        # Models dir: to store the trained models.
        self.model_dir = self.log_dir + 'models/'
        os.makedirs(self.model_dir, exist_ok=False)

        self.logger.info('Model folder {} created.'.format(self.model_dir))

    def add_file_handler_to_logger(self, logfile: str) -> None:
        """
        Add a ``logging.FileHandler`` to the logger.

        Specifies a ``logging.Formatter``:
            >>> logging.Formatter(fmt='[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
            ...                   datefmt='%Y-%m-%d %H:%M:%S')

        :param logfile: File used by the ``FileHandler``.

        """
        # create file handler which logs even DEBUG messages
        fh = logging.FileHandler(logfile)

        # set logging level for this file
        fh.setLevel(logging.DEBUG)

        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            fmt='[%(asctime)s] - %(levelname)s - %(name)s >>> %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)

        # add the handler to the logger
        self.logger.addHandler(fh)

    def initialize_statistics_collection(self) -> None:
        """
        Initializes 2 :py:class:`StatisticsCollector` to track statistics for training and validation.

        Adds some default statistics, such as the loss, episode idx and the epoch idx.

        Also creates the output files (csv).
        """
        # TRAINING.
        # Create statistics collector for training.
        self.training_stat_col = StatisticsCollector()

        # add default statistics
        self.training_stat_col.add_statistic('epoch', '{:02d}')
        self.training_stat_col.add_statistic('loss', '{:12.10f}')
        self.training_stat_col.add_statistic('episode', '{:06d}')
        self.training_stat_col.add_statistic('src_seq_length', '{:02d}')

        # Create the csv file to store the training statistics.
        self.training_batch_stats_file = self.training_stat_col.initialize_csv_file(
            self.log_dir,
            'training_statistics.csv')

        # VALIDATION.
        # Create statistics collector for validation.
        self.validation_stat_col = StatisticsCollector()

        # add default statistics
        self.validation_stat_col.add_statistic('epoch', '{:02d}')
        self.validation_stat_col.add_statistic('loss', '{:12.10f}')
        self.validation_stat_col.add_statistic('episode', '{:06d}')

        # Create the csv file to store the validation statistics.
        self.validation_batch_stats_file = self.validation_stat_col.initialize_csv_file(
            self.log_dir,
            'validation_statistics.csv')

    def finalize_statistics_collection(self) -> None:
        """
        Finalizes the statistics collection by closing the csv files.
        """
        # Close all files.
        self.training_batch_stats_file.close()
        self.validation_batch_stats_file.close()

    def initialize_tensorboard(self, log_dir = None) -> None:
        """
        Initializes the TensorBoard writers, and log directories.
        """
        from tensorboardX import SummaryWriter

        if log_dir is None:
            log_dir = self.log_dir

        self.training_writer = SummaryWriter(join(log_dir, "tensorboard", 'training'))
        self.training_stat_col.initialize_tensorboard(self.training_writer)

        self.validation_writer = SummaryWriter(join(log_dir, "tensorboard", 'validation'))
        self.validation_stat_col.initialize_tensorboard(self.validation_writer)

    def finalize_tensorboard(self) -> None:
        """
        Finalizes the operation of TensorBoard writers by closing them.
        """
        # Close the TensorBoard writers.
        self.training_writer.close()
        self.validation_writer.close()

    def set_random_seeds(self, pytorch_seed: int, numpy_seed: int, random_seed: int) -> None:
        """
        Set all random seeds to ensure the reproducibility of the experiments.
        Notably:

        - Set the random seed of Pytorch (seed the RNG for all devices (both CPU and CUDA):

            >>> torch.manual_seed(pytorch_seed)

        - When running on the CuDNN backend, two further options must be set:

            >>> torch.backends.cudnn.deterministic = True
            >>> torch.backends.cudnn.benchmark = False

        - Set the random seed of numpy:

            >>> np.random.seed(numpy_seed)

        - Finally, we initialize the random number generator:

            >>> random.seed(random_seed)

        """

        # set pytorch seed
        torch.manual_seed(pytorch_seed)

        # set deterministic CuDNN
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # set numpy seed
        import numpy
        numpy.random.seed(numpy_seed)

        # set the state of the random number generator:
        random.seed(random_seed)

        self.logger.info("torch seed was set to {}".format(pytorch_seed))
        self.logger.info("numpy seed was set to {}".format(numpy_seed))
        self.logger.info("random seed was set to {}".format(random_seed))


def save_model(job_dir, filepath, model_name):
    """Saves the model to Google Cloud Storage"""
    # Example: job_dir = 'gs://BUCKET_ID/hptuning_sonar/1'
    job_dir = job_dir.replace('gs://', '')  # Remove the 'gs://'
    # Get the Bucket Id
    bucket_id = job_dir.split('/')[0]
    # Get the path. Example: 'hptuning_sonar/1'
    bucket_path = job_dir.lstrip('{}/'.format(bucket_id))

    # Upload the model to GCS
    bucket = storage.Client().bucket(bucket_id)
    blob = bucket.blob('{}/{}'.format(
        bucket_path,
        model_name))
    blob.upload_from_filename(filepath)


if __name__ == '__main__':
    params = {
        "training": {
            "epochs": 3,
            "train_batch_size": 1024,
            "valid_batch_size": 1024,
            "smoothing": 0.1,
            "load_trained_model": False,
            "trained_model_checkpoint": ""
        },

        "settings": {
            "pytorch_seed": 0,
            "numpy_seed": 0,
            "random_seed": 0,
            "save_intermediate": False,
            "multi_gpu": True
        },

        "optim": {
            "lr": 0.,
            "betas": (0.9, 0.98),
            "eps": 1e-9,
            "factor": 1,
            "warmup": 2000,
            "step": 0

        },

        "dataset": {
            "max_seq_length": 40,  # ~ 90% of the training set
            "min_freq": 2,
            "start_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<blank>"

        },

        "model": {
            'd_model': 512,
            'N': 6,
            'dropout': 0.1,

            'attention': {
                'n_head': 8,
                'd_k': 64,
                'd_v': 64,
                'dropout': 0.1},

            'feed-forward': {
                'd_ff': 2048,
                'dropout': 0.1}
        }
    }
    trainer = Trainer(params)
    trainer.train()

    # Try to predict the following sequence:
    # first sentence in the validation dataset
    batch = next(iter(IWSLTDatasetBuilder.masked(
        IWSLTDatasetBuilder.transposed(trainer.validation_dataset_iterator))))

    if torch.cuda.is_available():
        batch.cuda()

    if trainer.multi_gpu:
        prediction = trainer.model.module.greedy_decode(batch.src[0].unsqueeze(0),
                                                        batch.src_mask[0], trainer.trg_vocab,
                                                        start_symbol="<s>", stop_symbol="</s>",
                                                        max_length=15)
    else:
        prediction = trainer.model.greedy_decode(batch.src[0].unsqueeze(0), batch.src_mask[0],
                                                 trainer.trg_vocab, start_symbol="<s>",
                                                 stop_symbol="</s>",
                                                 max_length=params["dataset"]["max_seq_length"])

    target, target_sentence = "", batch.trg[0]
    for i in target_sentence:
        target += trainer.trg_vocab.itos[i] + " "

    print("Trying to predict: {}".format(target))
    print("Got: {}".format(prediction))
