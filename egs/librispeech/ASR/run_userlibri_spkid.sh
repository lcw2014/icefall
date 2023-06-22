#!/usr/bin/env bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

dl_dir=$PWD/download


stage=10

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
stop_stage=100

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: train per spk id"
  mkdir -p rnn_lm/exp_averaged
  if [ ! -f rnn_lm/exp_averaged/epoch-30.pt ]; then cp rnn_lm/exp/epoch-30.pt rnn_lm/exp_averaged; fi
  filename="data/id_to_books.txt"

  for epoch in 35 40 45 50 55 60 65 70 75 80; do
    result_path="pruned_transducer_stateless5/results_fullFT_spkid_"$epoch".txt"

    if [ -f $result_path ]; then rm $result_path; fi
    while read line; do
      # echo $line
      spk_id=$(echo "$line" | cut -f1)
      book_ids=$(echo "$line" | cut -f2-)

      num=$(echo $book_ids | grep -o -E "[0-9]+")

      if [ ! -d rnn_lm/exp_$spk_id ]; then mkdir -p rnn_lm/exp_$spk_id; fi
      cp rnn_lm/exp/epoch-30.pt rnn_lm/exp_"$spk_id"
      ./rnn_lm/train_plm_perid.py \
        --start-epoch $((epoch-4)) \
        --world-size 3 \
        --num-epochs $((epoch+1)) \
        --use-fp16 0 \
        --embedding-dim 2048 \
        --hidden-dim 2048 \
        --num-layers 3 \
        --batch-size 100 \
        --exp-dir rnn_lm/exp_$spk_id \
        --save-last-epoch true \
        --lm-data-path data/lm_training_bpe_500_userlibri \
        --lm-data-name "$num"

      if [ -f rnn_lm/exp_$spk_id/epoch-30.pt ]; then rm rnn_lm/exp_$spk_id/epoch-30.pt; fi

      ./pruned_transducer_stateless5/decode_userlibri_perid.py \
        --epoch 30 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
        --max-duration 400 \
        --decoding-method modified_beam_search_lm_shallow_fusion \
        --beam-size 4 \
        --num-encoder-layer 12 \
        --lm-type rnn \
        --lm-scale 0.15 \
        --lm-exp-dir rnn_lm/exp_$spk_id \
        --lm-epoch $epoch \
        --lm-avg 1 \
        --rnn-lm-num-layers 3 \
        --use-shallow-fusion 1 \
        --rnn-lm-tie-weights 1 \
        --test-id "$book_ids" \
        --spk-id "$spk_id" \
        --result-path $result_path
    done < "$filename"
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Stage 2: train and average, finetuning for 10epoch"
  mkdir -p rnn_lm/exp_averaged
  if [ ! -f rnn_lm/exp_averaged/epoch-30.pt ]; then cp rnn_lm/exp/epoch-30.pt rnn_lm/exp_averaged; fi
  filename="data/id_to_books.txt"

  for epoch in 40 50 60; do
    result_path=pruned_transducer_stateless5/results_baseline_spkid_"$epoch".txt
    result_path2=pruned_transducer_stateless5/results_baseline_spkid_"$epoch"_avg.txt

    if [ -f $result_path ]; then rm $result_path; fi
    if [ -f $result_path2 ]; then rm $result_path2; fi

    while read line; do
      spk_id=$(echo "$line" | cut -f1)
      book_ids=$(echo "$line" | cut -f2-)

      num=$(echo $book_ids | grep -o -E "[0-9]+")
      # echo $line
      if [ ! -d rnn_lm/exp_$spk_id ]; then mkdir -p rnn_lm/exp_$spk_id; fi
      cp rnn_lm/exp_averaged/epoch-$((epoch-10)).pt rnn_lm/exp_$spk_id/epoch-$((epoch-10)).pt

      ./rnn_lm/train_plm_perid.py \
        --start-epoch $((epoch-9)) \
        --world-size 3 \
        --num-epochs $((epoch+1)) \
        --use-fp16 0 \
        --embedding-dim 2048 \
        --hidden-dim 2048 \
        --num-layers 3 \
        --batch-size 100 \
        --exp-dir rnn_lm/exp_$spk_id \
        --save-last-epoch true \
        --lm-data-path data/lm_training_bpe_500_userlibri \
        --lm-data-name "$num" \
      
      # if [ "$epoch" != 40 ]; then
      #   if [ -f rnn_lm/exp_$id/epoch-$((epoch-5)).pt ]; then rm rnn_lm/exp_$id/epoch-$((epoch-5)).pt; fi
      # fi
      if [ -f rnn_lm/exp_$spk_id/epoch-30.pt ]; then rm rnn_lm/exp_$spk_id/epoch-30.pt; fi
      ./pruned_transducer_stateless5/decode_userlibri_perid.py \
        --epoch 30 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
        --max-duration 400 \
        --decoding-method modified_beam_search_lm_shallow_fusion \
        --beam-size 4 \
        --num-encoder-layer 12 \
        --lm-type rnn \
        --lm-scale 0.15 \
        --lm-exp-dir rnn_lm/exp_$spk_id \
        --lm-epoch $epoch \
        --lm-avg 1 \
        --rnn-lm-num-layers 3 \
        --use-shallow-fusion 1 \
        --rnn-lm-tie-weights 1 \
        --test-id "$book_ids" \
        --spk-id "$spk_id" \
        --result-path $result_path
    done < "$filename"

    lm_list=pruned_transducer_stateless5/results_baseline_spkid_"$epoch".txt
    ./utils/average_parameters_spk.py \
      --lm-list ${lm_list} \
      --start-epoch $((epoch+1)) \
      --exp-dir rnn_lm/exp_averaged
    
    while read line; do
      spk_id=$(echo "$line" | cut -f1)
      book_ids=$(echo "$line" | cut -f2-)
      num=$(echo $book_ids | grep -o -E "[0-9]+")
      ./pruned_transducer_stateless5/decode_userlibri_perid.py \
        --epoch 30 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
        --max-duration 400 \
        --decoding-method modified_beam_search_lm_shallow_fusion \
        --beam-size 4 \
        --num-encoder-layer 12 \
        --lm-type rnn \
        --lm-scale 0.15 \
        --lm-exp-dir rnn_lm/exp_averaged \
        --lm-epoch $epoch \
        --lm-avg 1 \
        --rnn-lm-num-layers 3 \
        --use-shallow-fusion 1 \
        --rnn-lm-tie-weights 1 \
        --test-id "$book_ids" \
        --spk-id "$spk_id" \
        --result-path $result_path2
    done < "$filename"
  done
  echo "Stage 2-1: overfitting stage"

  for epoch in 65 70 75 80; do
    
    result_path=pruned_transducer_stateless5/results_baseline_spkid_overfitting_"$epoch".txt

    if [ -f $result_path ]; then rm $result_path; fi
    while read line; do
      spk_id=$(echo "$line" | cut -f1)
      book_ids=$(echo "$line" | cut -f2-)
      num=$(echo $book_ids | grep -o -E "[0-9]+")

      cp rnn_lm/exp_averaged/epoch-60.pt rnn_lm/exp_$spk_id/epoch-60.pt

      ./rnn_lm/train_plm_perid.py \
        --start-epoch $((epoch-4)) \
        --world-size 3 \
        --num-epochs $((epoch+1)) \
        --use-fp16 0 \
        --embedding-dim 2048 \
        --hidden-dim 2048 \
        --num-layers 3 \
        --batch-size 100 \
        --exp-dir rnn_lm/exp_$spk_id \
        --save-last-epoch true \
        --lm-data-path data/lm_training_bpe_500_userlibri \
        --lm-data-name "$num"
      
      ./pruned_transducer_stateless5/decode_userlibri_perid.py \
        --epoch 30 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
        --max-duration 400 \
        --decoding-method modified_beam_search_lm_shallow_fusion \
        --beam-size 4 \
        --num-encoder-layer 12 \
        --lm-type rnn \
        --lm-scale 0.15 \
        --lm-exp-dir rnn_lm/exp_$spk_id \
        --lm-epoch $epoch \
        --lm-avg 1 \
        --rnn-lm-num-layers 3 \
        --use-shallow-fusion 1 \
        --rnn-lm-tie-weights 1 \
        --test-id "$book_ids" \
        --spk-id "$spk_id" \
        --result-path $result_path
    done < "$filename"
  done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Stage 3: train per spk id with MAML fed, finetuning for 10epoch"
  mkdir -p rnn_lm/exp_fed
  if [ ! -f rnn_lm/exp_fed/epoch-30.pt ]; then cp rnn_lm/exp/epoch-30.pt rnn_lm/exp_fed; fi
  filename="data/id_to_books.txt"
  alpha=(1e-4)
  beta=(1e-1)

  for i in "${!alpha[@]}"; do
    for epoch in 40 50 60; do
      result_path=pruned_transducer_stateless5/results_per_spkid_fed_"$epoch"_${alpha[$i]}_${beta[$i]}.txt
      result_path2=pruned_transducer_stateless5/results_per_spkid_fed_"$epoch"_${alpha[$i]}_${beta[$i]}_avg.txt

      if [ -f $result_path ]; then rm $result_path; fi
      if [ -f $result_path2 ]; then rm $result_path2; fi
      echo ${alpha[$i]}_${beta[$i]}
      while read line; do
        spk_id=$(echo "$line" | cut -f1)
        book_ids=$(echo "$line" | cut -f2-)

        num=$(echo $book_ids | grep -o -E "[0-9]+")
        # echo $line
        if [ ! -d rnn_lm/exp_$spk_id ]; then mkdir -p rnn_lm/exp_$spk_id; fi
        cp rnn_lm/exp_fed/epoch-$((epoch-10)).pt rnn_lm/exp_$spk_id/epoch-$((epoch-10)).pt
        

        ./rnn_lm/train_plm_fed_perid.py \
          --start-epoch $((epoch-9)) \
          --world-size 3 \
          --num-epochs $((epoch+1)) \
          --use-fp16 0 \
          --embedding-dim 2048 \
          --hidden-dim 2048 \
          --num-layers 3 \
          --batch-size 100 \
          --exp-dir rnn_lm/exp_$spk_id \
          --save-last-epoch true \
          --lm-data-path data/lm_training_bpe_500_userlibri \
          --lm-data-name "$num" \
          --alpha ${alpha[$i]} \
          --beta ${beta[$i]}
        
        # if [ "$epoch" != 40 ]; then
        #   if [ -f rnn_lm/exp_$id/epoch-$((epoch-5)).pt ]; then rm rnn_lm/exp_$id/epoch-$((epoch-5)).pt; fi
        # fi

        if [ -f rnn_lm/exp_$spk_id/epoch-30.pt ]; then rm rnn_lm/exp_$spk_id/epoch-30.pt; fi
        ./pruned_transducer_stateless5/decode_userlibri_perid.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_$spk_id \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$book_ids" \
          --spk-id "$spk_id" \
          --result-path $result_path
          
      done < "$filename"

      lm_list=pruned_transducer_stateless5/results_per_spkid_fed_"$epoch"_${alpha[$i]}_${beta[$i]}.txt
      ./utils/average_parameters_spk.py \
        --lm-list ${lm_list} \
        --start-epoch $((epoch+1)) \
        --exp-dir rnn_lm/exp_fed
      
      while read line; do
        spk_id=$(echo "$line" | cut -f1)
        book_ids=$(echo "$line" | cut -f2-)
        num=$(echo $book_ids | grep -o -E "[0-9]+")
        ./pruned_transducer_stateless5/decode_userlibri_perid.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_fed \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$book_ids" \
          --spk-id "$spk_id" \
          --result-path $result_path2
      done < "$filename"
    done

    echo "Stage 23-1: overfitting stage"

    for epoch in 65 70 75 80; do
      
      result_path=pruned_transducer_stateless5/results_per_spkid_fed_overfitting_"$epoch".txt

      if [ -f $result_path ]; then rm $result_path; fi
      while read line; do
        spk_id=$(echo "$line" | cut -f1)
        book_ids=$(echo "$line" | cut -f2-)
        num=$(echo $book_ids | grep -o -E "[0-9]+")
        cp rnn_lm/exp_fed/epoch-60.pt rnn_lm/exp_$spk_id/epoch-60.pt

        ./rnn_lm/train_plm_perid.py \
          --start-epoch $((epoch-4)) \
          --world-size 3 \
          --num-epochs $((epoch+1)) \
          --use-fp16 0 \
          --embedding-dim 2048 \
          --hidden-dim 2048 \
          --num-layers 3 \
          --batch-size 100 \
          --exp-dir rnn_lm/exp_$book_spk_idid \
          --save-last-epoch true \
          --lm-data-path data/lm_training_bpe_500_userlibri \
          --lm-data-name "$num"
        
        ./pruned_transducer_stateless5/decode_userlibri_perid.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_$book_id \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$book_ids" \
          --spk-id "$spk_id" \
          --result-path $result_path
      done < "$filename"
    done
  done
fi