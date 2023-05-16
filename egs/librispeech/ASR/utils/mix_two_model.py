#!/usr/bin/env python3

import torch
import torch.nn as nn
from rnn_lm.model import RnnLmModel
import argparse
import logging
import math
from pathlib import Path
from shutil import copyfile
from typing import Optional, Tuple
import sys

import torch.multiprocessing as mp
import torch.optim as optim
from lhotse.utils import fix_random_seed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from icefall.checkpoint import load_checkpoint
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import AttributeDict, MetricsTracker, setup_logger, str2bool

import copy

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="""Resume training from from this epoch.
        If it is positive, it will load checkpoint from
        exp_dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="rnn_lm/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, logs, etc, are saved
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=True,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--lm-data",
        type=str,
        default="data/lm_training_bpe_500/sorted_lm_data.pt",
        help="LM training data",
    )

    parser.add_argument(
        "--lm-data-valid",
        type=str,
        default="data/lm_training_bpe_500/sorted_lm_data-valid.pt",
        help="LM validation data",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=500,
        help="Vocabulary size of the model",
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=2048,
        help="Embedding dim of the model",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=2048,
        help="Hidden dim of the model",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of RNN layers the model",
    )

    parser.add_argument(
        "--tie-weights",
        type=str2bool,
        default=True,
        help="""True to share the weights between the input embedding layer and the
        last output linear layer
        """,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--train-n-layer",
        type=int,
        default=0,
        help="freeze rnn layer",
    )

    parser.add_argument(
        "--surplus-layer",
        type=bool,
        default=False,
        help="add surplus rnn layer to existing rnn layers",
    )

    parser.add_argument(
        "--copy-last-layer",
        type=bool,
        default=False,
        help="add surplus rnn layer to existing rnn layers",
    )

    parser.add_argument(
        "--adapter",
        type=bool,
        default=False,
        help="add surplus rnn layer to existing rnn layers",
    )

    parser.add_argument(
        "--save-last-epoch",
        type=bool,
        default=False,
        help="add surplus rnn layer to existing rnn layers",
    )

    parser.add_argument(
        "--model-paths",
        type=str,
        help="LM training data",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Hidden dim of the model",
    )

    return parser

def get_params() -> AttributeDict:
    """Return a dict containing training parameters."""

    params = AttributeDict(
        {
            "max_sent_len": 200,
            "sos_id": 1,
            "eos_id": 1,
            "blank_id": 0,
            "lr": 1e-3,
            "weight_decay": 1e-6,
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 200,
            "reset_interval": 2000,
            "valid_interval": 5000,
            "env_info": get_env_info(),
        }
    )
    return params

def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    id: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:
    """Load checkpoint from file.

    If params.start_epoch is positive, it will load the checkpoint from
    `params.start_epoch - 1`. Otherwise, this function does nothing.

    Apart from loading state dict for `model`, `optimizer` and `scheduler`,
    it also updates `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The learning rate scheduler we are using.
    Returns:
      Return None.
    """
    if params.start_epoch <= 0:
        return

    filename = id

    logging.info(f"Loading checkpoint: {filename}")
    saved_params = load_checkpoint(
        filename,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # saved_params = saved_params['model']
    # print(saved_params)
    # keys = [
    #     "best_train_epoch",
    #     "best_valid_epoch",
    #     "batch_idx_train",
    #     "best_train_loss",
    #     "best_valid_loss",
    # ]
    # for k in keys:
    #     params[k] = saved_params[k]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
    """
    if rank != 0:
        return
    import os
    if not os.path.exists(params.exp_dir):
        os.makedirs(params.exp_dir)
    filename = f"{params.exp_dir}/epoch-{params.start_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        rank=rank,
    )


def average_models(params, models, weights):
    """
    Averages a list of PyTorch models with corresponding weights
    Args:
        models (list): List of PyTorch models
        weights (list): List of weights for each model
    Returns:
        avg_model (nn.Module): Averaged model
    """
    # Checking the number of models and weights
    assert len(models) == len(weights), "Number of models and weights must match"
    
    # Get the device for the first model
    device = next(models[0].parameters()).device

    # Create a new model with the same structure as the input models
    avg_model = RnnLmModel(
            vocab_size=params.vocab_size,
            embedding_dim=params.embedding_dim,
            hidden_dim=params.hidden_dim,
            num_layers=params.num_layers,
            tie_weights=params.tie_weights,
            surplus_layer=params.surplus_layer,
            adapter=params.adapter,
        )
    for avg_param in avg_model.parameters():
        avg_param.data.fill_(0)

    # Copy the weights from each model into the averaged model
    for model, weight in zip(models, weights):
        for avg_param, param in zip(avg_model.parameters(), model.parameters()):
            avg_param.data.add_(weight * param.data)
    
    return avg_model

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))
    device = torch.device("cpu")

    model_paths = params.model_paths.split()
    models = list()
    print(model_paths)
    weights = torch.tensor([params.alpha, 1 - params.alpha]).float()

    from copy import deepcopy
    for did in model_paths:
        model = RnnLmModel(
            vocab_size=params.vocab_size,
            embedding_dim=params.embedding_dim,
            hidden_dim=params.hidden_dim,
            num_layers=params.num_layers,
            tie_weights=params.tie_weights,
            surplus_layer=params.surplus_layer,
            adapter=params.adapter,
        )
        _ = load_checkpoint_if_available(params=params, model=model, id=did)

        model.to(device)
        model.device = device
        models.append(deepcopy(model))
    model = average_models(params, models, weights=weights)

    save_checkpoint(
        params=params,
        model=model,
        optimizer=None,
    )
    
    