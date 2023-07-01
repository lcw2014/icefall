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
  mkdir -p transformer_lm/exp_averaged
  if [ ! -f transformer_lm/exp_averaged/epoch-12.pt ]; then cp transformer_lm/exp/epoch-12.pt transformer_lm/exp_averaged; fi
  filename="data/id_to_books.txt"

  for epoch in 17 22 27 32 37 42 47 52; do
    result_path="pruned_transducer_stateless5/results_fullFT_spkid_"$epoch".txt"

    if [ -f $result_path ]; then rm $result_path; fi
    while read line; do
      echo $line
      spk_id=$(echo "$line" | cut -f1)
      book_ids=$(echo "$line" | cut -f2-)

      num=$(echo $book_ids | grep -o -E "[0-9]+")

      if [ ! -d transformer_lm/exp_$spk_id ]; then mkdir -p transformer_lm/exp_$spk_id; fi
      cp transformer_lm/exp/epoch-12.pt transformer_lm/exp_"$spk_id"
      ./transformer_lm/train_plm_perid.py \
        --start-epoch $((epoch-4)) \
        --world-size 3 \
        --num-epochs $((epoch+1)) \
        --use-fp16 0 \
        --num-layers 16 \
        --batch-size 100 \
        --exp-dir transformer_lm/exp_$spk_id \
        --save-last-epoch true \
        --lm-data-path data/lm_training_bpe_500_userlibri \
        --lm-data-name "$spk_id"

      ./pruned_transducer_stateless5/decode_userlibri_perid.py \
        --epoch 30 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
        --max-duration 400 \
        --num-buckets 2 \
        --decoding-method modified_beam_search_lm_shallow_fusion \
        --beam-size 4 \
        --lm-type transformer \
        --lm-scale 0.15 \
        --num-encoder-layer 12 \
        --lm-exp-dir transformer_lm/exp_$spk_id \
        --transformer-lm-num-layers 16 \
        --transformer-lm-tie-weights 1 \
        --use-shallow-fusion 1 \
        --lm-epoch $epoch \
        --test-id "$book_ids" \
        --spk-id "$spk_id" \
        --result-path $result_path
      if [ -f transformer_lm/exp_$spk_id/epoch-12.pt ]; then rm transformer_lm/exp_$spk_id/epoch-12.pt; fi

    done < "$filename"
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Stage 2: train and average, finetuning for 10epoch"
  mkdir -p transformer_lm/exp_averaged
  if [ ! -f transformer_lm/exp_averaged/epoch-12.pt ]; then cp transformer_lm/exp/epoch-12.pt transformer_lm/exp_averaged; fi
  filename="data/id_to_books.txt"

  for epoch in 22 32 42; do
    result_path=pruned_transducer_stateless5/results_baseline_spkid_"$epoch".txt
    result_path2=pruned_transducer_stateless5/results_baseline_spkid_"$epoch"_avg.txt

    # if [ -f $result_path ]; then rm $result_path; fi
    # if [ -f $result_path2 ]; then rm $result_path2; fi

    while read line; do
      spk_id=$(echo "$line" | cut -f1)
      book_ids=$(echo "$line" | cut -f2-)
      if [ $epoch -eq 22 ]; then break; fi
      num=$(echo $book_ids | grep -o -E "[0-9]+")
      # echo $line
      if [ ! -d transformer_lm/exp_$spk_id ]; then mkdir -p transformer_lm/exp_$spk_id; fi
      cp transformer_lm/exp_averaged/epoch-$((epoch-10)).pt transformer_lm/exp_$spk_id/epoch-$((epoch-10)).pt

      ./transformer_lm/train_plm_perid.py \
        --start-epoch $((epoch-9)) \
        --world-size 3 \
        --num-epochs $((epoch+1)) \
        --use-fp16 0 \
        --num-layers 16 \
        --batch-size 90 \
        --exp-dir transformer_lm/exp_$spk_id \
        --save-last-epoch true \
        --lm-data-path data/lm_training_bpe_500_userlibri \
        --lm-data-name "$spk_id"
      
      # if [ "$epoch" != 40 ]; then
      #   if [ -f transformer_lm/exp_$id/epoch-$((epoch-5)).pt ]; then rm transformer_lm/exp_$id/epoch-$((epoch-5)).pt; fi
      # fi
      if [ -f transformer_lm/exp_$spk_id/epoch-12.pt ]; then rm transformer_lm/exp_$spk_id/epoch-12.pt; fi
      ./pruned_transducer_stateless5/decode_userlibri_perid.py \
        --epoch 30 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
        --max-duration 400 \
        --num-buckets 2 \
        --decoding-method modified_beam_search_lm_shallow_fusion \
        --beam-size 4 \
        --lm-type transformer \
        --lm-scale 0.15 \
        --num-encoder-layer 12 \
        --lm-exp-dir transformer_lm/exp_$spk_id \
        --transformer-lm-num-layers 16 \
        --transformer-lm-tie-weights 1 \
        --use-shallow-fusion 1 \
        --lm-epoch $epoch \
        --test-id "$book_ids" \
        --spk-id "$spk_id" \
        --result-path $result_path
    done < "$filename"

    lm_list=pruned_transducer_stateless5/results_baseline_spkid_"$epoch".txt
    ./utils/average_parameters_spk_TF.py \
      --lm-list ${lm_list} \
      --start-epoch $((epoch+1)) \
      --exp-dir transformer_lm/exp_averaged
    
    while read line; do
      spk_id=$(echo "$line" | cut -f1)
      book_ids=$(echo "$line" | cut -f2-)
      num=$(echo $book_ids | grep -o -E "[0-9]+")
      ./pruned_transducer_stateless5/decode_userlibri_perid.py \
        --epoch 30 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
        --max-duration 400 \
        --num-buckets 2 \
        --decoding-method modified_beam_search_lm_shallow_fusion \
        --beam-size 4 \
        --lm-type transformer \
        --lm-scale 0.15 \
        --num-encoder-layer 12 \
        --lm-exp-dir transformer_lm/exp_averaged \
        --transformer-lm-num-layers 16 \
        --transformer-lm-tie-weights 1 \
        --use-shallow-fusion 1 \
        --lm-epoch $epoch \
        --test-id "$book_ids" \
        --spk-id "$spk_id" \
        --result-path $result_path2
    done < "$filename"
  done
  echo "Stage 2-1: overfitting stage"

  for epoch in 47 52; do
    
    result_path3=pruned_transducer_stateless5/results_baseline_spkid_overfitting_"$epoch".txt

    if [ -f $result_path ]; then rm $result_path; fi
    while read line; do
      spk_id=$(echo "$line" | cut -f1)
      book_ids=$(echo "$line" | cut -f2-)
      num=$(echo $book_ids | grep -o -E "[0-9]+")

      cp transformer_lm/exp_averaged/epoch-42.pt transformer_lm/exp_$spk_id/epoch-42.pt

      ./transformer_lm/train_plm_perid.py \
        --start-epoch $((epoch-4)) \
        --world-size 3 \
        --num-epochs $((epoch+1)) \
        --use-fp16 0 \
        --num-layers 16 \
        --batch-size 90 \
        --exp-dir transformer_lm/exp_$spk_id \
        --save-last-epoch true \
        --lm-data-path data/lm_training_bpe_500_userlibri \
        --lm-data-name "$spk_id"
      
      ./pruned_transducer_stateless5/decode_userlibri_perid.py \
        --epoch 30 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
        --max-duration 400 \
        --num-buckets 2 \
        --decoding-method modified_beam_search_lm_shallow_fusion \
        --beam-size 4 \
        --lm-type transformer \
        --lm-scale 0.15 \
        --num-encoder-layer 12 \
        --lm-exp-dir transformer_lm/exp_$spk_id \
        --transformer-lm-num-layers 16 \
        --transformer-lm-tie-weights 1 \
        --use-shallow-fusion 1 \
        --lm-epoch $epoch \
        --test-id "$book_ids" \
        --spk-id "$spk_id" \
        --result-path $result_path3
    done < "$filename"
  done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Stage 3: train per spk id with MAML fed, finetuning for 10epoch"
  mkdir -p transformer_lm/exp_fed
  if [ ! -f transformer_lm/exp_fed/epoch-12.pt ]; then cp transformer_lm/exp/epoch-12.pt transformer_lm/exp_fed; fi
  filename="data/id_to_books.txt"
  alpha=(1e-4)
  beta=(1e-1)

  for i in "${!alpha[@]}"; do
    for epoch in 22 32 42; do
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
        if [ ! -d transformer_lm/exp_$spk_id ]; then mkdir -p transformer_lm/exp_$spk_id; fi
        cp transformer_lm/exp_fed/epoch-$((epoch-10)).pt transformer_lm/exp_$spk_id/epoch-$((epoch-10)).pt
        

        ./transformer_lm/train_plm_fed_perid.py \
          --start-epoch $((epoch-9)) \
          --world-size 3 \
          --num-epochs $((epoch+1)) \
          --use-fp16 0 \
          --num-layers 16 \
          --batch-size 90 \
          --exp-dir transformer_lm/exp_$spk_id \
          --save-last-epoch true \
          --lm-data-path data/lm_training_bpe_500_userlibri \
          --lm-data-name "$spk_id" \
          --alpha ${alpha[$i]} \
          --beta ${beta[$i]}
        
        # if [ "$epoch" != 40 ]; then
        #   if [ -f transformer_lm/exp_$id/epoch-$((epoch-5)).pt ]; then rm transformer_lm/exp_$id/epoch-$((epoch-5)).pt; fi
        # fi

        if [ -f transformer_lm/exp_$spk_id/epoch-12.pt ]; then rm transformer_lm/exp_$spk_id/epoch-12.pt; fi
        ./pruned_transducer_stateless5/decode_userlibri_perid.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --num-buckets 2 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --lm-type transformer \
          --lm-scale 0.15 \
          --num-encoder-layer 12 \
          --lm-exp-dir transformer_lm/exp_$spk_id \
          --transformer-lm-num-layers 16 \
          --transformer-lm-tie-weights 1 \
          --use-shallow-fusion 1 \
          --lm-epoch $epoch \
          --test-id "$book_ids" \
          --spk-id "$spk_id" \
          --result-path $result_path
        exit
      done < "$filename"

      lm_list=pruned_transducer_stateless5/results_per_spkid_fed_"$epoch"_${alpha[$i]}_${beta[$i]}.txt
      ./utils/average_parameters_spk_TF.py \
        --lm-list ${lm_list} \
        --start-epoch $((epoch+1)) \
        --exp-dir transformer_lm/exp_fed
      
      while read line; do
        spk_id=$(echo "$line" | cut -f1)
        book_ids=$(echo "$line" | cut -f2-)
        num=$(echo $book_ids | grep -o -E "[0-9]+")
        ./pruned_transducer_stateless5/decode_userlibri_perid.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --num-buckets 2 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --lm-type transformer \
          --lm-scale 0.15 \
          --num-encoder-layer 12 \
          --lm-exp-dir transformer_lm/exp_fed \
          --transformer-lm-num-layers 16 \
          --transformer-lm-tie-weights 1 \
          --use-shallow-fusion 1 \
          --lm-epoch $epoch \
          --test-id "$book_ids" \
          --spk-id "$spk_id" \
          --result-path $result_path2
      done < "$filename"
    done

    echo "Stage 3-1: overfitting stage"

    for epoch in 47 52; do
      
      result_path3=pruned_transducer_stateless5/results_per_spkid_fed_overfitting_"$epoch".txt

      if [ -f $result_path ]; then rm $result_path; fi
      while read line; do
        spk_id=$(echo "$line" | cut -f1)
        book_ids=$(echo "$line" | cut -f2-)
        num=$(echo $book_ids | grep -o -E "[0-9]+")
        cp transformer_lm/exp_fed/epoch-42.pt transformer_lm/exp_$spk_id/epoch-42.pt

        ./transformer_lm/train_plm_perid.py \
          --start-epoch $((epoch-4)) \
          --world-size 3 \
          --num-epochs $((epoch+1)) \
          --use-fp16 0 \
          --num-layers 16 \
          --batch-size 90 \
          --exp-dir transformer_lm/exp_$spk_id \
          --save-last-epoch true \
          --lm-data-path data/lm_training_bpe_500_userlibri \
          --lm-data-name "$spk_id"
        
        ./pruned_transducer_stateless5/decode_userlibri_perid.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --num-buckets 2 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --lm-type transformer \
          --lm-scale 0.15 \
          --num-encoder-layer 12 \
          --lm-exp-dir transformer_lm/exp_$spk_id \
          --transformer-lm-num-layers 16 \
          --transformer-lm-tie-weights 1 \
          --use-shallow-fusion 1 \
          --lm-epoch $epoch \
          --test-id "$book_ids" \
          --spk-id "$spk_id" \
          --result-path $result_path3
      done < "$filename"
    done
  done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "Stage 4: train per spk id with MAML fed, finetuning for 10epoch, model selection"
  mkdir -p transformer_lm/exp_fed
  if [ ! -f transformer_lm/exp_fed/epoch-12.pt ]; then cp transformer_lm/exp/epoch-12.pt transformer_lm/exp_fed; fi
  filename="data/id_to_books.txt"
  alpha=(1e-4)
  beta=(1e-1)

  for i in "${!alpha[@]}"; do
    for epoch in 22 32 42; do
      result_path=pruned_transducer_stateless5/results_per_spkid_fed2_"$epoch"_${alpha[$i]}_${beta[$i]}.txt
      result_path2=pruned_transducer_stateless5/results_per_spkid_fed2_"$epoch"_${alpha[$i]}_${beta[$i]}_avg.txt

      if [ -f $result_path ]; then rm $result_path; fi
      if [ -f $result_path2 ]; then rm $result_path2; fi
      echo ${alpha[$i]}_${beta[$i]}
      while read line; do
        spk_id=$(echo "$line" | cut -f1)
        book_ids=$(echo "$line" | cut -f2-)

        num=$(echo $book_ids | grep -o -E "[0-9]+")
        # echo $line
        if [ ! -d transformer_lm/exp_$spk_id ]; then mkdir -p transformer_lm/exp_$spk_id; fi
        cp transformer_lm/exp_fed/epoch-$((epoch-10)).pt transformer_lm/exp_$spk_id/epoch-$((epoch-10)).pt
        

        ./transformer_lm/train_plm_fed_perid.py \
          --start-epoch $((epoch-9)) \
          --world-size 3 \
          --num-epochs $((epoch+1)) \
          --use-fp16 0 \
          --num-layers 16 \
          --batch-size 90 \
          --exp-dir transformer_lm/exp_$spk_id \
          --save-last-epoch true \
          --lm-data-path data/lm_training_bpe_500_userlibri \
          --lm-data-name "$spk_id" \
        
        # if [ "$epoch" != 40 ]; then
        #   if [ -f transformer_lm/exp_$id/epoch-$((epoch-5)).pt ]; then rm transformer_lm/exp_$id/epoch-$((epoch-5)).pt; fi
        # fi

        if [ -f transformer_lm/exp_$spk_id/epoch-12.pt ]; then rm transformer_lm/exp_$spk_id/epoch-12.pt; fi
        ./pruned_transducer_stateless5/decode_userlibri_perid.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --num-buckets 2 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --lm-type transformer \
          --lm-scale 0.15 \
          --num-encoder-layer 12 \
          --lm-exp-dir transformer_lm/exp_$spk_id \
          --transformer-lm-num-layers 16 \
          --transformer-lm-tie-weights 1 \
          --use-shallow-fusion 1 \
          --lm-epoch $epoch \
          --test-id "$book_ids" \
          --spk-id "$spk_id" \
          --result-path $result_path
        
        # model selection
        if (( $epoch > 22 )); then
          fname1=pruned_transducer_stateless5/results_per_spkid_fed2_"$epoch"_${alpha[$i]}_${beta[$i]}.txt
          fname2=pruned_transducer_stateless5/results_per_spkid_fed2_$((epoch-10))_${alpha[$i]}_${beta[$i]}.txt
          temp_id=userlibri-$spk_id
          while IFS=$'\t' read -r id value; do
            echo $id $value
            if [ "$id" == "$temp_id" ]; then
              value_file1="$value"
              echo $value_file1
              break
            fi
          done < "$fname1"

          # Read file 2 and find the value for the specific ID
          while IFS=$'\t' read -r id value; do
            if [ "$id" == "$temp_id" ]; then
              value_file2="$value"
              break
            fi
          done < "$fname2"

          if (( $(echo "$value_file1 < $value_file2" | bc -l) )); then
            echo model selected
            rm transformer_lm/exp_$spk_id/epoch-"$epoch".pt && cp transformer_lm/exp_$spk_id/epoch-$((epoch-10)).pt transformer_lm/exp_$spk_id/epoch-"$epoch".pt
          fi
        fi
      done < "$filename"

      lm_list=pruned_transducer_stateless5/results_per_spkid_fed2_"$epoch"_${alpha[$i]}_${beta[$i]}.txt
      ./utils/average_parameters_spk_TF.py \
        --lm-list ${lm_list} \
        --start-epoch $((epoch+1)) \
        --exp-dir transformer_lm/exp_fed
      
      while read line; do
        spk_id=$(echo "$line" | cut -f1)
        book_ids=$(echo "$line" | cut -f2-)
        num=$(echo $book_ids | grep -o -E "[0-9]+")
        ./pruned_transducer_stateless5/decode_userlibri_perid.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --num-buckets 2 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --lm-type transformer \
          --lm-scale 0.15 \
          --num-encoder-layer 12 \
          --lm-exp-dir transformer_lm/exp_fed \
          --transformer-lm-num-layers 16 \
          --transformer-lm-tie-weights 1 \
          --use-shallow-fusion 1 \
          --lm-epoch $epoch \
          --test-id "$book_ids" \
          --spk-id "$spk_id" \
          --result-path $result_path2
      done < "$filename"
    done

    echo "Stage 4-1: overfitting stage"

    for epoch in 47 52; do
      
      result_path3=pruned_transducer_stateless5/results_per_spkid_fed2_overfitting_"$epoch".txt

      if [ -f $result_path ]; then rm $result_path; fi
      while read line; do
        spk_id=$(echo "$line" | cut -f1)
        book_ids=$(echo "$line" | cut -f2-)
        num=$(echo $book_ids | grep -o -E "[0-9]+")
        cp transformer_lm/exp_fed/epoch-42.pt transformer_lm/exp_$spk_id/epoch-42.pt

        ./transformer_lm/train_plm_perid.py \
          --start-epoch $((epoch-4)) \
          --world-size 3 \
          --num-epochs $((epoch+1)) \
          --use-fp16 0 \
          --num-layers 16 \
          --batch-size 90 \
          --exp-dir transformer_lm/exp_$spk_id \
          --save-last-epoch true \
          --lm-data-path data/lm_training_bpe_500_userlibri \
          --lm-data-name "$spk_id"
        
        ./pruned_transducer_stateless5/decode_userlibri_perid.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --num-buckets 2 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --lm-type transformer \
          --lm-scale 0.15 \
          --num-encoder-layer 12 \
          --lm-exp-dir transformer_lm/exp_$spk_id \
          --transformer-lm-num-layers 16 \
          --transformer-lm-tie-weights 1 \
          --use-shallow-fusion 1 \
          --lm-epoch $epoch \
          --test-id "$book_ids" \
          --spk-id "$spk_id" \
          --result-path $result_path3
      done < "$filename"
    done
  done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  echo "Stage 5: train per spk id with MAML fed, finetuning for 10epoch, model convex combination"
  mkdir -p transformer_lm/exp_fed
  if [ ! -f transformer_lm/exp_fed/epoch-12.pt ]; then cp transformer_lm/exp/epoch-12.pt transformer_lm/exp_fed; fi
  filename="data/id_to_books.txt"
  alpha=(1e-4)
  beta=(1e-1)

  for i in "${!alpha[@]}"; do
    for epoch in 22 32 42; do
      result_path=pruned_transducer_stateless5/results_per_spkid_fed3_"$epoch"_${alpha[$i]}_${beta[$i]}.txt
      result_path2=pruned_transducer_stateless5/results_per_spkid_fed3_"$epoch"_${alpha[$i]}_${beta[$i]}_avg.txt

      if [ -f $result_path ]; then rm $result_path; fi
      if [ -f $result_path2 ]; then rm $result_path2; fi
      echo ${alpha[$i]}_${beta[$i]}
      while read line; do
        spk_id=$(echo "$line" | cut -f1)
        book_ids=$(echo "$line" | cut -f2-)

        num=$(echo $book_ids | grep -o -E "[0-9]+")
        echo $line
        if [ ! -d transformer_lm/exp_$spk_id ]; then mkdir -p transformer_lm/exp_$spk_id; fi
        cp transformer_lm/exp_fed/epoch-$((epoch-10)).pt transformer_lm/exp_$spk_id/epoch-$((epoch-10)).pt
        

        ./transformer_lm/train_plm_fed_perid.py \
          --start-epoch $((epoch-9)) \
          --world-size 3 \
          --num-epochs $((epoch+1)) \
          --use-fp16 0 \
          --num-layers 16 \
          --batch-size 90 \
          --exp-dir transformer_lm/exp_$spk_id \
          --save-last-epoch true \
          --lm-data-path data/lm_training_bpe_500_userlibri \
          --lm-data-name "$spk_id" \
        
        # if [ "$epoch" != 40 ]; then
        #   if [ -f transformer_lm/exp_$id/epoch-$((epoch-5)).pt ]; then rm transformer_lm/exp_$id/epoch-$((epoch-5)).pt; fi
        # fi

        
        ./pruned_transducer_stateless5/decode_userlibri_perid.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --num-buckets 2 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --lm-type transformer \
          --lm-scale 0.15 \
          --num-encoder-layer 12 \
          --lm-exp-dir transformer_lm/exp_$spk_id \
          --transformer-lm-num-layers 16 \
          --transformer-lm-tie-weights 1 \
          --use-shallow-fusion 1 \
          --lm-epoch $epoch \
          --test-id "$book_ids" \
          --spk-id "$spk_id" \
          --result-path $result_path
        
        # convex combination
        result_path3=pruned_transducer_stateless5/dummy/results_"$spk_id"_"$epoch"_convex.txt
        if [ -f $result_path3 ]; then rm $result_path3; fi

        for alpha2 in $(seq 0 0.1 1); do
          ./utils/mix_two_model.py \
            --model-paths "transformer_lm/exp_$spk_id/epoch-$epoch.pt transformer_lm/exp_$spk_id/epoch-$((epoch-10)).pt" \
            --exp-dir transformer_lm/exp_averaged \
            --alpha ${alpha2}
          
          ./pruned_transducer_stateless5/decode_userlibri_perid.py \
            --epoch 30 \
            --avg 15 \
            --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
            --max-duration 400 \
            --num-buckets 2 \
            --decoding-method modified_beam_search_lm_shallow_fusion \
            --beam-size 4 \
            --lm-type transformer \
            --lm-scale 0.15 \
            --num-encoder-layer 12 \
            --lm-exp-dir transformer_lm/exp_averaged \
            --transformer-lm-num-layers 16 \
            --transformer-lm-tie-weights 1 \
            --use-shallow-fusion 1 \
            --lm-epoch $epoch \
            --test-id "$book_ids" \
            --spk-id "$spk_id" \
            --result-path $result_path3
        done
        lowest_value=50
        line_number=0
        lowest_line_number=0
        while IFS=$'\t' read -r id value; do
          line_number=$((line_number+1))
          # Check if the value is lower than the lowest value found so far
          result=$(echo "$value <= $lowest_value" | bc -l)
          if [[ "$result" -eq 1 ]]; then
            lowest_value=$value
            lowest_line_number=$line_number
          fi
        done < "$result_path3"
        number=0
        for alpha2 in $(seq 0 0.1 1); do
          number=$((number+1))
          if [[ $number == $lowest_line_number ]]; then
            echo "FOUND OPTIMAL POINT"
            echo "$number $lowest_line_number $alpha2 $spk_id $alpha2"
            rm transformer_lm/exp_$spk_id/epoch-$epoch.pt && cp transformer_lm/exp_averaged/epoch-$alpha2.pt transformer_lm/exp_$spk_id/epoch-$epoch.pt
            break
          fi
        done
        if [ -f transformer_lm/exp_$spk_id/epoch-12.pt ]; then rm transformer_lm/exp_$spk_id/epoch-12.pt; fi
      done < "$filename"

      lm_list=pruned_transducer_stateless5/results_per_spkid_fed3_"$epoch"_${alpha[$i]}_${beta[$i]}.txt
      ./utils/average_parameters_spk_TF.py \
        --lm-list ${lm_list} \
        --start-epoch $((epoch+1)) \
        --exp-dir transformer_lm/exp_fed
      
      while read line; do
        spk_id=$(echo "$line" | cut -f1)
        book_ids=$(echo "$line" | cut -f2-)
        num=$(echo $book_ids | grep -o -E "[0-9]+")
        ./pruned_transducer_stateless5/decode_userlibri_perid.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --num-buckets 2 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --lm-type transformer \
          --lm-scale 0.15 \
          --num-encoder-layer 12 \
          --lm-exp-dir transformer_lm/exp_fed \
          --transformer-lm-num-layers 16 \
          --transformer-lm-tie-weights 1 \
          --use-shallow-fusion 1 \
          --lm-epoch $epoch \
          --test-id "$book_ids" \
          --spk-id "$spk_id" \
          --result-path $result_path2
      done < "$filename"
    done

    echo "Stage 5-1: overfitting stage"

    for epoch in 47 52; do
      
      result_path4=pruned_transducer_stateless5/results_per_spkid_fed3_overfitting_"$epoch".txt

      if [ -f $result_path ]; then rm $result_path; fi
      while read line; do
        spk_id=$(echo "$line" | cut -f1)
        book_ids=$(echo "$line" | cut -f2-)
        num=$(echo $book_ids | grep -o -E "[0-9]+")
        cp transformer_lm/exp_fed/epoch-42.pt transformer_lm/exp_$spk_id/epoch-42.pt

        ./transformer_lm/train_plm_perid.py \
          --start-epoch $((epoch-4)) \
          --world-size 3 \
          --num-epochs $((epoch+1)) \
          --use-fp16 0 \
          --num-layers 16 \
          --batch-size 90 \
          --exp-dir transformer_lm/exp_$spk_id \
          --save-last-epoch true \
          --lm-data-path data/lm_training_bpe_500_userlibri \
          --lm-data-name "$spk_id"
        
        ./pruned_transducer_stateless5/decode_userlibri_perid.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --num-buckets 2 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --lm-type transformer \
          --lm-scale 0.15 \
          --num-encoder-layer 12 \
          --lm-exp-dir transformer_lm/exp_$spk_id \
          --transformer-lm-num-layers 16 \
          --transformer-lm-tie-weights 1 \
          --use-shallow-fusion 1 \
          --lm-epoch $epoch \
          --test-id "$book_ids" \
          --spk-id "$spk_id" \
          --result-path $result_path4
      done < "$filename"
    done
  done
fi


# if [ $stage -le 99 ] && [ $stop_stage -ge 99 ]; then
#     echo "Stage 100: Generate LM training data"

#   for vocab_size in ${vocab_sizes[@]}; do
#     echo "Processing vocab_size == ${vocab_size}"
#     lang_dir=data/lang_bpe_${vocab_size}
#     out_dir=data/lm_training_bpe_${vocab_size}_userlibri
#     mkdir -p $out_dir
#     filename="data/id_to_books.txt"
#     while read line; do
#       spk_id=$(echo "$line" | cut -f1)
#       book_ids=$(echo "$line" | cut -f2-)
      
#       num=$(echo $book_ids | grep -o -E "[0-9]+")
#       python3 ./local/prepare_lm_training_data_spkid.py \
#         --bpe-model $lang_dir/bpe.model \
#         --lm-data "$num" \
#         --lm-archive $out_dir/lm_data_$spk_id.pt \
#         --out-dir $out_dir
#     done < "$filename"
#   done
# fi

# if [ $stage -le 100 ] && [ $stop_stage -ge 100 ]; then
#   echo "Stage 6: Sort LM training data"
#   # Sort LM training data by sentence length in descending order
#   # for ease of training.
#   #
#   # Sentence length equals to the number of BPE tokens
#   # in a sentence.

#   for vocab_size in ${vocab_sizes[@]}; do
#     out_dir=data/lm_training_bpe_${vocab_size}_userlibri
#     mkdir -p $out_dir
#     filename="data/id_to_books.txt"
#     while read line; do
#         spk_id=$(echo "$line" | cut -f1)
#         book_ids=$(echo "$line" | cut -f2-)
        
#         num=$(echo $book_ids | grep -o -E "[0-9]+")
#         ./local/sort_lm_training_data.py \
#         --in-lm-data $out_dir/lm_data_$spk_id.pt \
#         --out-lm-data $out_dir/sorted_lm_data_$spk_id.pt \
#         --out-statistics $out_dir/statistics_$spk_id.txt
#     done < "$filename"
#   done
# fi
