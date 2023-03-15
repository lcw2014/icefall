(8) modified beam search with RNNLM shallow fusion
./pruned_transducer_stateless5/decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
    --max-duration 1200 \
    --decoding-method modified_beam_search_lm_shallow_fusion \
    --beam-size 4 \
    --lm-type transformer \
    --lm-scale 0.3 \
    --num-encoder-layer 12 \
    --lm-exp-dir /home/Workspace/icefall/icefall/transformer_lm/exp \
    --transformer-lm-num-layers 16 \
    --transformer-lm-tie-weights 1 \
    --use-shallow-fusion 1 \
    --lm-epoch 12

./pruned_transducer_stateless5/decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
    --max-duration 600 \
    --decoding-method modified_beam_search_lm_shallow_fusion \
    --beam-size 4 \
    --num-encoder-layer 12 \
    --lm-type rnn \
    --lm-scale 0.3 \
    --lm-exp-dir icefall/egs/librispeech/ASR/rnn_lm/exp_10136/epoch-34.pt \
    --lm-epoch 34 \
    --lm-avg 1 \
    --rnn-lm-num-layers 3 \
    --use-shallow-fusion 1 \
    --rnn-lm-tie-weights 1

./pruned_transducer_stateless5/decode_userlibri.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
    --max-duration 400 \
    --decoding-method modified_beam_search_lm_shallow_fusion \
    --beam-size 4 \
    --num-encoder-layer 12 \
    --lm-type rnn \
    --lm-scale 0.3 \
    --lm-exp-dir rnn_lm/exp_10136 \
    --lm-epoch 34 \
    --lm-avg 1 \
    --rnn-lm-num-layers 3 \
    --use-shallow-fusion 1 \
    --rnn-lm-tie-weights 1 \
    --test-id test-clean_2300

./pruned_transducer_stateless_d2v_v2/decode.py \
    --input-strategy AudioSamples \
    --enable-spec-aug False \
    --additional-block True \
    --model-name epoc.pt \
    --exp-dir ./pruned_transducer_stateless_d2v_v2/exp \
    --max-duration 400 \
    --decoding-method modified_beam_search_lm_shallow_fusion \
    --max-sym-per-frame 1 \ 
    --encoder-type d2v \
    --encoder-dim 768 \
    --decoder-dim 768 \
    --joiner-dim 768 \
