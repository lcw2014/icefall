#!/usr/bin/env bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

dl_dir=$PWD/download


stage=10
stop_stage=100

parts=(
    test-clean
    test-other
)
. shared/parse_options.sh || exit 1

vocab_sizes=(
  # 5000
  # 2000
  # 1000
  500
)
export PYTHONPATH=/home/lee/Workspace/icefall

# ./pruned_transducer_stateless5_kd/decode.py \
#   --epoch 30 \
#   --avg 15 \
#   --exp-dir pruned_transducer_stateless5/exp_layer12 \
#   --max-duration 400 \
#   --decoding-method modified_beam_search_lm_shallow_fusion \
#   --beam-size 4 \
#   --num-encoder-layer 12 \
#   --lm-type rnn \
#   --lm-scale 0.3 \
#   --lm-exp-dir rnn_lm/exp \
#   --lm-epoch 30 \
#   --lm-avg 1 \
#   --rnn-lm-num-layers 3 \
#   --use-shallow-fusion 1 \
#   --rnn-lm-tie-weights 1

# ./pruned_transducer_stateless5_kd/decode.py \
#   --epoch 30 \
#   --avg 1 \
#   --exp-dir pruned_transducer_stateless5_kd/exp_ver2 \
#   --max-duration 400 \
#   --decoding-method modified_beam_search_lm_clm_shallow_fusion \
#   --beam-size 4 \
#   --num-encoder-layer 12 \
#   --lm-type rnn \
#   --lm-scale 0.3 \
#   --lm-exp-dir pruned_transducer_stateless5_kd/rnn_lm/exp \
#   --lm-epoch 30 \
#   --lm-avg 1 \
#   --rnn-lm-num-layers 3 \
#   --use-shallow-fusion 1 \
#   --rnn-lm-tie-weights 1
# exit
# ./pruned_transducer_stateless5_kd/decode.py \
#   --epoch 40 \
#   --avg 15 \
#   --exp-dir pruned_transducer_stateless5_kd/exp_ver2 \
#   --max-duration 400 \
#   --decoding-method modified_beam_search_lm_clm_shallow_fusion \
#   --beam-size 4 \
#   --num-encoder-layer 12 \
#   --lm-type rnn \
#   --lm-scale 0.3 \
#   --lm-exp-dir pruned_transducer_stateless5_kd/rnn_lm/exp \
#   --lm-epoch 35 \
#   --lm-avg 1 \
#   --rnn-lm-num-layers 3 \
#   --use-shallow-fusion 1 \
#   --rnn-lm-tie-weights 1 || exit
# exit
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: "
  # export CUDA_VISIBLE_DEVICES="0,1,2,3"
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
  ./pruned_transducer_stateless5_kd/train.py \
    --world-size 4 \
    --num-epochs 30 \
    --start-epoch 13 \
    --exp-dir pruned_transducer_stateless5_kd/exp_ver2 \
    --num-encoder-layers 12 \
    --full-libri 1 \
    --max-duration 300 \
    --vocab-size 500 \
    --embedding-dim 2048 \
    --hidden-dim 2048 \
    --lm-num-layers 3 \
    --lm-epoch 30 \
    --num-clm-layers 3 \
    --lm-dir "pruned_transducer_stateless5_kd/rnn_lm/exp" \
    --nll-loss-scale 1 || exit
  
  ./pruned_transducer_stateless5_kd/decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir pruned_transducer_stateless5_kd/exp_ver2 \
    --max-duration 400 \
    --decoding-method modified_beam_search_lm_shallow_fusion \
    --beam-size 4 \
    --num-encoder-layer 12 \
    --lm-type rnn \
    --lm-scale 0.3 \
    --lm-exp-dir pruned_transducer_stateless5_kd/rnn_lm/exp \
    --lm-epoch 30 \
    --lm-avg 1 \
    --rnn-lm-num-layers 3 \
    --use-shallow-fusion 1 \
    --rnn-lm-tie-weights 1 || exit
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Stage 2: "
  # export CUDA_VISIBLE_DEVICES="0,1,2,3"
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
  # ./pruned_transducer_stateless5_kd/train_lm.py \
  #   --world-size 4 \
  #   --num-epochs 35 \
  #   --start-epoch 31 \
  #   --exp-dir pruned_transducer_stateless5_kd/exp_ver2 \
  #   --num-encoder-layers 12 \
  #   --full-libri 1 \
  #   --max-duration 600 \
  #   --vocab-size 500 \
  #   --embedding-dim 2048 \
  #   --hidden-dim 2048 \
  #   --lm-num-layers 3 \
  #   --lm-epoch 30 \
  #   --num-clm-layers 3 \
  #   --lm-training true \
  #   --lm-dir "pruned_transducer_stateless5_kd/rnn_lm/exp" \
  #   --nll-loss-scale 1 || exit
  
  ./pruned_transducer_stateless5_kd/train.py \
    --world-size 4 \
    --num-epochs 40 \
    --start-epoch 36 \
    --exp-dir pruned_transducer_stateless5_kd/exp_ver2 \
    --num-encoder-layers 12 \
    --full-libri 1 \
    --max-duration 300 \
    --vocab-size 500 \
    --embedding-dim 2048 \
    --hidden-dim 2048 \
    --lm-num-layers 3 \
    --lm-epoch 35 \
    --num-clm-layers 3 \
    --lm-training false \
    --lm-dir "pruned_transducer_stateless5_kd/rnn_lm/exp" \
    --nll-loss-scale 1 || exit
  
  ./pruned_transducer_stateless5_kd/decode.py \
    --epoch 40 \
    --avg 15 \
    --exp-dir pruned_transducer_stateless5_kd/exp_ver2 \
    --max-duration 400 \
    --decoding-method modified_beam_search_lm_shallow_fusion \
    --beam-size 4 \
    --num-encoder-layer 12 \
    --lm-type rnn \
    --lm-scale 0.3 \
    --lm-exp-dir pruned_transducer_stateless5_kd/rnn_lm/exp \
    --lm-epoch 30 \
    --lm-avg 1 \
    --rnn-lm-num-layers 3 \
    --use-shallow-fusion 1 \
    --rnn-lm-tie-weights 1 || exit
fi