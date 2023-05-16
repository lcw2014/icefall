#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Usage:
./rnn_lm/train_plm.py \
    --start-epoch 31 \
    --world-size 4 \
    --num-epochs 35 \
    --use-fp16 0 \
    --embedding-dim 2048 \
    --hidden-dim 2048 \
    --num-layers 3 \
    --batch-size 400 \
    --exp-dir rnn_lm/exp_10136 \
    --lm-data data/lm_training_bpe_500_userlibri/sorted_lm_data_10136.pt
"""

import argparse
import logging
import math
from pathlib import Path
from shutil import copyfile
from typing import Optional, Tuple, Union
import sys

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader, get_dataloader_perid, get_dataloader_fed
from lhotse.utils import fix_random_seed
from model_fed import RnnLmModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from icefall.checkpoint import load_checkpoint
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import AttributeDict, MetricsTracker, setup_logger, str2bool

import copy
from copy import deepcopy
from collections import OrderedDict
import torch.nn.functional as F
import random

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
        help="",
    )

    parser.add_argument(
        "--adapter",
        type=bool,
        default=False,
        help="",
    )

    parser.add_argument(
        "--save-last-epoch",
        type=bool,
        default=False,
        help="",
    )

    parser.add_argument(
        "--lm-data-name",
        type=str,
        help="",
    )

    parser.add_argument(
        "--lm-data-path",
        type=str,
        help="",
    )
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--beta", type=float, default=1e-3)

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

    filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    logging.info(f"Loading checkpoint: {filename}")
    saved_params = load_checkpoint(
        filename,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
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
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        rank=rank,
    )

    # if params.best_train_epoch == params.cur_epoch:
    #     best_train_filename = params.exp_dir / "best-train-loss.pt"
    #     copyfile(src=filename, dst=best_train_filename)

    # if params.best_valid_epoch == params.cur_epoch:
    #     best_valid_filename = params.exp_dir / "best-valid-loss.pt"
    #     copyfile(src=filename, dst=best_valid_filename)


def compute_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    sentence_lengths: torch.Tensor,
    is_training: bool,
) -> Tuple[torch.Tensor, MetricsTracker]:
    """Compute the negative log-likelihood loss given a model and its input.
    Args:
      model:
        The NN model, e.g., RnnLmModel.
      x:
        A 2-D tensor. Each row contains BPE token IDs for a sentence. Also,
        each row starts with SOS ID.
      y:
        A 2-D tensor. Each row is a shifted version of the corresponding row
        in `x` but ends with an EOS ID (before padding).
     sentence_lengths:
       A 1-D tensor containing number of tokens of each sentence
       before padding.
     is_training:
       True for training. False for validation.
    """
    with torch.set_grad_enabled(is_training):
        device = model.device
        x = x.to(device)
        y = y.to(device)
        sentence_lengths = sentence_lengths.to(device)

        nll = model(x, y, sentence_lengths)
        loss = nll.sum()

        num_tokens = sentence_lengths.sum().item()

        loss_info = MetricsTracker()
        # Note: Due to how MetricsTracker() is designed,
        # we use "frames" instead of "num_tokens" as a key here
        loss_info["frames"] = num_tokens + 1
        loss_info["loss"] = loss.detach().item()
    return loss, loss_info


def compute_validation_loss(
    params: AttributeDict,
    model: nn.Module,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process. The validation loss
    is saved in `params.valid_loss`.
    """
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        x, y, sentence_lengths = batch
        with torch.cuda.amp.autocast(enabled=params.use_fp16):
            loss, loss_info = compute_loss(
                model=model,
                x=x,
                y=y,
                sentence_lengths=sentence_lengths,
                is_training=False,
            )

        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss

def gradient_clip_(
    grads: torch.Tensor,
):
    total_norm = torch.norm(torch.stack([torch.norm(grad.detach(), 2.0) for grad in grads]), 2.0)
    clip_coef = 1.0 / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    # print(total_norm,clip_coef_clamped)
    for grad in grads:
        grad.mul_(clip_coef_clamped.to(grad.device))

def compute_grad(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    sentence_lengths: torch.Tensor,
    is_training: bool,
    params: AttributeDict,
    v: Union[Tuple[torch.Tensor, ...], None] = None,
    second_order_grads:bool = False,
) -> Tuple[torch.Tensor, MetricsTracker]:
    """Compute the negative log-likelihood loss given a model and its input.
    Args:
      model:
        The NN model, e.g., RnnLmModel.
      x:
        A 2-D tensor. Each row contains BPE token IDs for a sentence. Also,
        each row starts with SOS ID.
      y:
        A 2-D tensor. Each row is a shifted version of the corresponding row
        in `x` but ends with an EOS ID (before padding).
     sentence_lengths:
       A 1-D tensor containing number of tokens of each sentence
       before padding.
     is_training:
       True for training. False for validation.
    """
    with torch.set_grad_enabled(is_training):
        device = model.device
        x = x.to(device)
        y = y.to(device)
        sentence_lengths = sentence_lengths.to(device)

        if second_order_grads:
            frz_model_params = deepcopy(model.state_dict())
            delta = 1e-3
            dummy_model_params_1 = OrderedDict()
            dummy_model_params_2 = OrderedDict()
            with torch.no_grad():
                for (layer_name, param), grad in zip(model.named_parameters(), v):
                    dummy_model_params_1.update({layer_name: param + delta * grad})
                    dummy_model_params_2.update({layer_name: param - delta * grad})
            
            model.load_state_dict(dummy_model_params_1, strict=False)
            loss_1 = model(x, y, sentence_lengths)
            # loss_1 = F.cross_entropy(
            # logit_1.reshape(-1, params.vocab_size), y.reshape(-1), reduction="none"
            # )
            loss_1 = loss_1.sum()
            grads_1 = torch.autograd.grad(loss_1, model.parameters())

            gradient_clip_(grads_1)

            model.load_state_dict(dummy_model_params_2, strict=False)
            loss_2 = model(x, y, sentence_lengths)
            # loss_2 = F.cross_entropy(
            # logit_2.reshape(-1, params.vocab_size), y.reshape(-1), reduction="none"
            # )
            loss_2 = loss_2.sum()
            grads_2 = torch.autograd.grad(loss_2, model.parameters())

            gradient_clip_(grads_2)

            model.load_state_dict(frz_model_params)

            grads = []
            with torch.no_grad():
                for g1, g2 in zip(grads_1, grads_2):
                    grads.append((g1 - g2) / (2 * delta))
            
            loss_info = MetricsTracker()
            num_tokens = sentence_lengths.sum().item()
            loss_info["frames"] = num_tokens + 1
            loss_info["loss"] = loss_1.detach().item()



            # print(loss_1,loss_2)
            # print(loss_1.mean(),loss_2.mean())
            # print(loss_1.max(),loss_2.max())
            # if loss_1.isnan() or loss_2.isnan():
            #     exit()
            return grads, loss_info

        else:
            loss = model(x,y, sentence_lengths)
            # loss = F.cross_entropy(
            # logit.reshape(-1, params.vocab_size), y.reshape(-1), reduction="none"
            # )
            loss = loss.sum()
            grads = torch.autograd.grad(loss, model.parameters())
            gradient_clip_(grads)
    
            return grads
        # loss = nll.sum()

        # num_tokens = sentence_lengths.sum().item()

        # loss_info = MetricsTracker()
        # # Note: Due to how MetricsTracker() is designed,
        # # we use "frames" instead of "num_tokens" as a key here
        # loss_info["frames"] = num_tokens
        # loss_info["loss"] = loss.detach().item()

def train_one_epoch(
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dl: zip,
    valid_dl: torch.utils.data.DataLoader,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all sentences is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
    """
    model.train()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(train_dl):
        batch1, batch2, batch3 = batch[0], batch[1], batch[2]
        params.batch_idx_train += 1
        x1, y1, sentence_lengths1 = batch1
        x2, y2, sentence_lengths2 = batch2
        x3, y3, sentence_lengths3 = batch3
        # print(batch1)
        # print(batch2)
        # print(batch3)
        batch_size = x1.size(0)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=params.use_fp16):
            temp_model = deepcopy(model)
            grads = compute_grad(
                model=temp_model,
                x=x1,
                y=y1,
                sentence_lengths=sentence_lengths1,
                is_training=True,
                params=params,
            )
            for param, grad in zip(temp_model.parameters(), grads):
                param.data.sub_(params.alpha * grad)
            grads_1st = compute_grad(
                model=temp_model,
                x=x2,
                y=y2,
                sentence_lengths=sentence_lengths2,
                is_training=True,
                params=params,
            )

            grads_2nd, loss_info = compute_grad(
                model=model,
                x=x3,
                y=y3,
                sentence_lengths=sentence_lengths3,
                is_training=True,
                params=params,
                v=grads_1st,
                second_order_grads=True,
            )
            
            for param, grad1, grad2 in zip(
                model.parameters(), grads_1st, grads_2nd
            ):
                param.grad = params.beta * grad1 - params.beta * params.alpha * grad2
                # param.data.sub_(params.beta * grad1 - params.beta * params.alpha * grad2)

        # summary stats
        tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

        
        clip_grad_norm_(model.parameters(), 5.0, 2.0)
        optimizer.step()

        if batch_idx % params.log_interval == 0:
            # print(x3)
            # print(y3)
            # # Note: "frames" here means "num_tokens"
            # print(loss_info["loss"], loss_info["frames"])
            try:
                this_batch_ppl = math.exp(loss_info["loss"] / loss_info["frames"])
                tot_ppl = math.exp(tot_loss["loss"] / tot_loss["frames"])
            except OverflowError:
                this_batch_ppl = math.inf
                tot_ppl = math.inf

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}, ppl: {this_batch_ppl}] "
                f"tot_loss[{tot_loss}, ppl: {tot_ppl}], "
                f"batch size: {batch_size}"
            )

            if tb_writer is not None:
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)

                tb_writer.add_scalar(
                    "train/current_ppl", this_batch_ppl, params.batch_idx_train
                )

                tb_writer.add_scalar("train/tot_ppl", tot_ppl, params.batch_idx_train)

        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            logging.info("Computing validation loss")

            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            try:
                valid_ppl = math.exp(valid_info["loss"] / valid_info["frames"])
            except OverflowError:
                valid_ppl = math.inf
            logging.info(
                f"Epoch {params.cur_epoch}, validation: {valid_info}, "
                f"ppl: {valid_ppl}"
            )

            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

                tb_writer.add_scalar(
                    "train/valid_ppl", valid_ppl, params.batch_idx_train
                )
    # print(tot_loss.keys())
    # print(tot_loss["loss"])
    # print(tot_loss["frames"])
    try:
        loss_value = tot_loss["loss"] / tot_loss["frames"]
    except OverflowError:
        loss_value = math.inf

    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss
    
def train_n_layer(model, args):
    if args.train_n_layer != 4:
        for name, param in model.named_parameters():
            if list(name)[-1] != str(args.train_n_layer - 1):
                param.requires_grad_(False)

def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))
    is_distributed = world_size > 1

    fix_random_seed(params.seed)
    if is_distributed:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")
    logging.info(params)

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)

    logging.info(f"Device: {device}")

    logging.info("About to create model")

    model = RnnLmModel(
        vocab_size=params.vocab_size,
        embedding_dim=params.embedding_dim,
        hidden_dim=params.hidden_dim,
        num_layers=params.num_layers,
        tie_weights=params.tie_weights,
        surplus_layer=params.surplus_layer,
        adapter=params.adapter,
    )

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    checkpoints = load_checkpoint_if_available(params=params, model=model)

    if params.copy_last_layer:
        state_dict = model.state_dict()
        new_state_dict = copy.deepcopy(model.state_dict())
        for k in state_dict.keys():
            if list(k)[-1] == str(params.num_layers -1):
                name = '_'.join(k.split('.')[1].split('_')[0:2])
                new_state_dict[f'rnn_surplus_layer.{name}_l0'] = state_dict[k]
        
        model.load_state_dict(new_state_dict)
        if args.train_n_layer == 4:
            for name, param in model.named_parameters():
                if name.split('.')[0] != 'rnn_surplus_layer':
                    param.requires_grad_(False)
    
    if params.adapter:
        for n, p in model.named_parameters():
            if n.split('.')[0] != 'adapter':
                p.requires_grad_(False)

    model.to(device)
    if params.train_n_layer:
        train_n_layer(model,args)

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    model.device = device

    optimizer = optim.Adam(
        model.parameters(),
        lr=params.lr,
        weight_decay=params.weight_decay,
    )
    if checkpoints.get('optimizer',None):
        logging.info("Load optimizer state_dict from checkpoint")
        optimizer.load_state_dict(checkpoints["optimizer"])

    logging.info(f"Loading LM training data from {params.lm_data_name}")
    train_dl = get_dataloader_fed(
        filename=params.lm_data_name,
        is_distributed=is_distributed,
        params=params,
    )

    order1 = list(range(train_dl.dataset.__len__()))
    order2 = list(order1)
    order3 = list(order2)
    while any(order2[i] == order1[i] for i in range(len(order1))):
        random.shuffle(order2)
    while any(order3[i] == order1[i] or order3[i] == order2[i] for i in range(len(order1))):
        random.shuffle(order3)
    
    train_dl2 = get_dataloader_fed(
        filename=params.lm_data_name,
        is_distributed=is_distributed,
        params=params,
        order=order1
    )
    train_dl3 = get_dataloader_fed(
        filename=params.lm_data_name,
        is_distributed=is_distributed,
        params=params,
        order=order1
    )
    logging.info(f"Loading LM validation data from {params.lm_data_valid}")
    valid_dl = get_dataloader(
        filename=params.lm_data_valid,
        is_distributed=is_distributed,
        params=params,
    )

    # Note: No learning rate scheduler is used here
    for epoch in range(params.start_epoch, params.num_epochs):
        if is_distributed:
            train_dl.sampler.set_epoch(epoch)
            train_dl2.sampler.set_epoch(epoch)
            train_dl3.sampler.set_epoch(epoch)

        params.cur_epoch = epoch
        train_zip = zip(train_dl, train_dl2, train_dl3)
        train_one_epoch(
            params=params,
            model=model,
            optimizer=optimizer,
            train_dl=train_zip,
            valid_dl=valid_dl,
            tb_writer=tb_writer,
            world_size=world_size,
        )
        if params.save_last_epoch:
            if epoch == params.num_epochs -1:
                save_checkpoint(
                    params=params,
                    model=model,
                    optimizer=optimizer,
                    rank=rank,
                )
        else:
            save_checkpoint(
                params=params,
                model=model,
                optimizer=optimizer,
                rank=rank,
            )

    logging.info("Done!")

    if is_distributed:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
