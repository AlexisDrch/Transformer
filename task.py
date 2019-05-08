import argparse
import logging
import sys

import hypertune
import torch

import trainer as t


def get_args():
    """Argument parser.
    Returns:
        Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='PyTorch Trainer')
    parser.add_argument('--job-dir',  # handled automatically by AI Platform
                        help='GCS location to write checkpoints and export models')
    parser.add_argument('--is-test', default=False, action='store_true')
    parser.add_argument('--is-hyperparameter-tuning', default=False, action='store_true')
    parser.add_argument('--model-name',
                        type=str,
                        default="model.pt",
                        help='What to name the saved model file')
    parser.add_argument('--batch-size',
                        type=int,
                        default=48,
                        help='input batch size for training (default: 48)')
    parser.add_argument('--epochs',
                        type=int,
                        default=3,
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--beta0',  # Specified in the config file
                        type=float,
                        default=0.9,
                        help='Beta0 parameter (default: 0.9)')
    parser.add_argument('--beta1',  # Specified in the config file
                        type=float,
                        default=0.98,
                        help='Beta1 parameter (default: 0.98)')
    parser.add_argument('--warmup',  # Specified in the config file
                        type=int,
                        default=2000,
                        help='Warmup parameter (default: 2000)')
    parser.add_argument('--smoothing',  # Specified in the config file
                        type=float,
                        default=0.1,
                        help='Smoothing parameter (default: 0.1)')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed (default: 0)')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    params = {
        "training": {
            "epochs": args.epochs,
            "train_batch_size": args.batch_size,
            "valid_batch_size": args.batch_size,
            "smoothing": args.smoothing,
            "load_trained_model": False,
            "trained_model_checkpoint": ""
        },

        "settings": {
            "pytorch_seed": args.seed,
            "numpy_seed": args.seed,
            "random_seed": args.seed,
            "save_intermediate": False,
            "multi_gpu": True,
            "save_dir": args.job_dir,
            "model_name": args.model_name,
        },

        "optim": {
            "lr": 0.,
            "betas": (args.beta0, args.beta1),
            "eps": 1e-9,
            "factor": 1,
            "warmup": args.warmup,
            "step": 0,

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
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. This script is supposed to "
                        "run on a CUDA-enabled instance.")
        if args.is_test:
            sys.exit(0)
        else:
            sys.exit(1)

    if args.is_hyperparameter_tuning:
        logging.info("Hyper Parameter Tuning Mode")
        t.HYPERTUNER = hypertune.HyperTune()
    else:
        logging.info("Training Mode")

    trainer = t.Trainer(params)
    validation_loss = trainer.train()

    if not args.job_dir:
        print(f"Validation loss: {validation_loss}")
