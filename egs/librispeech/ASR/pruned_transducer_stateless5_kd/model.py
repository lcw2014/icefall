# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
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


from typing import Tuple

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_interface import EncoderInterface
from scaling import ScaledLinear

from icefall.utils import add_sos, add_eos, make_pad_mask
from rnn_lm.model import RnnLmModel


class CLTransducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
        clm: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and
            (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output
            contains unnormalized probs, i.e., not processed by log-softmax.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner
        self.clm = clm

        self.vocab_size = vocab_size

        self.simple_am_proj = ScaledLinear(encoder_dim, vocab_size, initial_speed=0.5)
        self.simple_lm_proj = ScaledLinear(decoder_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        lm_embeddings: torch.Tensor,
        lm_outputs: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        lm_training: bool = False,
        warmup: float = 1.0,
        reduction: str = "sum",
        delay_penalty: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
          warmup:
            A value warmup >= 0 that determines which modules are active, values
            warmup > 1 "are fully warmed up" and all modules will be active.
          reduction:
            "sum" to sum the losses over all utterances in the batch.
            "none" to return the loss in a 1-D tensor for each utterance
            in the batch.
          delay_penalty:
            A constant value used to penalize symbol delay, to encourage
            streaming models to emit symbols earlier.
            See https://github.com/k2-fsa/k2/issues/955 and
            https://arxiv.org/pdf/2211.00490.pdf for more details.
        Returns:
        Returns:
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert reduction in ("sum", "none"), reduction
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0
        if lm_training:
          with torch.no_grad():
            encoder_out, x_lens = self.encoder(x, x_lens, warmup=warmup)
        else:
          encoder_out, x_lens = self.encoder(x, x_lens, warmup=warmup)
        assert torch.all(x_lens > 0)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_id, eos_id = self.clm.sos_id, self.clm.eos_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        if not lm_training:
          decoder_out = self.decoder(sos_y_padded)

          # Note: y does not start with SOS
          # y_padded : [B, S]
          y_padded = y.pad(mode="constant", padding_value=0)

          y_padded = y_padded.to(torch.int64)
          boundary = torch.zeros((x.size(0), 4), dtype=torch.int64, device=x.device)
          boundary[:, 2] = y_lens
          boundary[:, 3] = x_lens

          lm = self.simple_lm_proj(decoder_out)
          am = self.simple_am_proj(encoder_out)
        # import sys
        # print(sos_y_padded.shape,y_padded.shape)
        # # torch.Size([3, 158]) torch.Size([3, 157])
        # print(sos_y_padded[0],y_padded[0])
        # # tensor([  0, 212, 364,   4, 249,   4,  80, 108,   3,   3,  16,  35,  79,  38,
        # # tensor([  212, 364,   4, 249,   4,  80, 108,   3,   3,  16,  35,  79,  38,
        # print(encoder_out.shape,am.shape)
        # # torch.Size([3, 774, 384]) torch.Size([3, 774, 500])
        # print(decoder_out.shape,lm.shape)
        # # torch.Size([3, 158, 512]) torch.Size([3, 158, 500])
        # sys.exit()
        y_lens += 1
        eos_y = add_eos(y, eos_id=blank_id)
        eos_y_padded = eos_y.pad(mode="constant", padding_value=blank_id).long()
        clm_out = self.clm(
          x=encoder_out,
          x_lens=x_lens,
          lm_embeddings=lm_embeddings,
          y=lm_outputs,
          y_lens=y_lens)
        nll_loss = F.cross_entropy(
            clm_out.reshape(-1, self.vocab_size), eos_y_padded.reshape(-1), reduction="none"
        )

        mask = make_pad_mask(y_lens).reshape(-1)
        nll_loss.masked_fill_(mask, 0)
        nll_loss = nll_loss.reshape(sos_y_padded.size(0), -1)
        if not lm_training:
          with torch.cuda.amp.autocast(enabled=False):
              simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                  lm=lm.float(),
                  am=am.float(),
                  symbols=y_padded,
                  termination_symbol=blank_id,
                  lm_only_scale=lm_scale,
                  am_only_scale=am_scale,
                  boundary=boundary,
                  reduction=reduction,
                  delay_penalty=delay_penalty,
                  return_grad=True,
              )

          # ranges : [B, T, prune_range]
          ranges = k2.get_rnnt_prune_ranges(
              px_grad=px_grad,
              py_grad=py_grad,
              boundary=boundary,
              s_range=prune_range,
          )

          # am_pruned : [B, T, prune_range, encoder_dim]
          # lm_pruned : [B, T, prune_range, decoder_dim]
          am_pruned, lm_pruned = k2.do_rnnt_pruning(
              am=self.joiner.encoder_proj(encoder_out),
              lm=self.joiner.decoder_proj(decoder_out),
              ranges=ranges,
          )

          # logits : [B, T, prune_range, vocab_size]

          # project_input=False since we applied the decoder's input projections
          # prior to do_rnnt_pruning (this is an optimization for speed).
          logits = self.joiner(am_pruned, lm_pruned, project_input=False)

          with torch.cuda.amp.autocast(enabled=False):
              pruned_loss = k2.rnnt_loss_pruned(
                  logits=logits.float(),
                  symbols=y_padded,
                  ranges=ranges,
                  termination_symbol=blank_id,
                  boundary=boundary,
                  delay_penalty=delay_penalty,
                  reduction=reduction,
              )
        if lm_training:
          return nll_loss
        else:
          return (simple_loss, pruned_loss, nll_loss)
    def compute_nll(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        lm_embeddings: torch.Tensor,
        lm_outputs: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        warmup: float = 1.0,
        reduction: str = "sum",
        delay_penalty: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
          warmup:
            A value warmup >= 0 that determines which modules are active, values
            warmup > 1 "are fully warmed up" and all modules will be active.
          reduction:
            "sum" to sum the losses over all utterances in the batch.
            "none" to return the loss in a 1-D tensor for each utterance
            in the batch.
          delay_penalty:
            A constant value used to penalize symbol delay, to encourage
            streaming models to emit symbols earlier.
            See https://github.com/k2-fsa/k2/issues/955 and
            https://arxiv.org/pdf/2211.00490.pdf for more details.
        Returns:
        Returns:
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert reduction in ("sum", "none"), reduction
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0
        with torch.no_grad():
          encoder_out, x_lens = self.encoder(x, x_lens, warmup=warmup)
        assert torch.all(x_lens > 0)

        # # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_id, eos_id = self.clm.sos_id, self.clm.eos_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        # decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        # y_padded = y.pad(mode="constant", padding_value=0)

        # y_padded = y_padded.to(torch.int64)
        # boundary = torch.zeros((x.size(0), 4), dtype=torch.int64, device=x.device)
        # boundary[:, 2] = y_lens
        # boundary[:, 3] = x_lens

        # lm = self.simple_lm_proj(decoder_out)
        # am = self.simple_am_proj(encoder_out)
        
        # import sys
        # print(sos_y_padded.shape,y_padded.shape)
        # # torch.Size([3, 158]) torch.Size([3, 157])
        # print(sos_y_padded[0],y_padded[0])
        # # tensor([  0, 212, 364,   4, 249,   4,  80, 108,   3,   3,  16,  35,  79,  38,
        # # tensor([  212, 364,   4, 249,   4,  80, 108,   3,   3,  16,  35,  79,  38,
        # print(encoder_out.shape,am.shape)
        # # torch.Size([3, 774, 384]) torch.Size([3, 774, 500])
        # print(decoder_out.shape,lm.shape)
        # # torch.Size([3, 158, 512]) torch.Size([3, 158, 500])
        # sys.exit()
        y_lens += 1
        eos_y = add_eos(y, eos_id=blank_id)
        eos_y_padded = eos_y.pad(mode="constant", padding_value=blank_id).long()
        clm_out = self.clm(
          x=encoder_out,
          x_lens=x_lens,
          lm_embeddings=lm_embeddings,
          y=lm_outputs,
          y_lens=y_lens)
        nll_loss = F.cross_entropy(
            clm_out.reshape(-1, self.vocab_size), eos_y_padded.reshape(-1), reduction="none"
        )
        mask = make_pad_mask(y_lens).reshape(-1)
        nll_loss.masked_fill_(mask, 0)
        nll_loss = nll_loss.reshape(sos_y_padded.size(0), -1)

        # with torch.cuda.amp.autocast(enabled=False):
        #     simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
        #         lm=lm.float(),
        #         am=am.float(),
        #         symbols=y_padded,
        #         termination_symbol=blank_id,
        #         lm_only_scale=lm_scale,
        #         am_only_scale=am_scale,
        #         boundary=boundary,
        #         reduction=reduction,
        #         delay_penalty=delay_penalty,
        #         return_grad=True,
        #     )

        # # ranges : [B, T, prune_range]
        # ranges = k2.get_rnnt_prune_ranges(
        #     px_grad=px_grad,
        #     py_grad=py_grad,
        #     boundary=boundary,
        #     s_range=prune_range,
        # )

        # # am_pruned : [B, T, prune_range, encoder_dim]
        # # lm_pruned : [B, T, prune_range, decoder_dim]
        # am_pruned, lm_pruned = k2.do_rnnt_pruning(
        #     am=self.joiner.encoder_proj(encoder_out),
        #     lm=self.joiner.decoder_proj(decoder_out),
        #     ranges=ranges,
        # )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        # logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        # with torch.cuda.amp.autocast(enabled=False):
        #     pruned_loss = k2.rnnt_loss_pruned(
        #         logits=logits.float(),
        #         symbols=y_padded,
        #         ranges=ranges,
        #         termination_symbol=blank_id,
        #         boundary=boundary,
        #         delay_penalty=delay_penalty,
        #         reduction=reduction,
        #     )

        return nll_loss
