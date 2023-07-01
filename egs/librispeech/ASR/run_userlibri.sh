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
stop_stage=$stage

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "stage 1: prepare manifest for userlibri"
    for part in ${parts[@]}; do
        gzip -d data/manifests/librispeech_supervisions_${part}.jsonl.gz
        cat data/manifests/librispeech_supervisions_${part}.jsonl > data/manifests/librispeech_supervisions_${part}_temp.jsonl
        gzip data/manifests/librispeech_supervisions_${part}.jsonl
        gzip -d data/manifests/librispeech_recordings_${part}.jsonl.gz
        cat data/manifests/librispeech_recordings_${part}.jsonl > data/manifests/librispeech_recordings_${part}_temp.jsonl
        gzip data/manifests/librispeech_recordings_${part}.jsonl
    done

    mkdir -p data/manifests
    python3 ./utils/prepare_userlibri.py
    # exit
    for part in ${parts[@]}; do
        rm data/manifests/librispeech_supervisions_${part}_temp.jsonl
        gzip data/manifests/userlibri_supervisions_${part}.jsonl
        rm data/manifests/librispeech_recordings_${part}_temp.jsonl
        gzip data/manifests/userlibri_recordings_${part}.jsonl
    done

fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Compute fbank for userlibri"

    mkdir -p data/fbank
    if [ ! -e data/fbank/.userlibri.done ]; then
        ./local/compute_fbank_userlibri.py
        exit
        touch data/fbank/.userlibri.done
    fi

    if [ ! -e data/fbank/.userlibri-validated.done ]; then
        echo "Validating data/fbank for userlibri"
        parts=$(cat data/manifests/validating_parts.txt)
        for part in ${parts[@]}; do
            python3 ./local/validate_manifest.py \
                data/fbank/userlibri_cuts_${part}.jsonl.gz
        done
        touch data/fbank/.userlibri-validated.done
    fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Generate LM training data"

    for vocab_size in ${vocab_sizes[@]}; do
        echo "Processing vocab_size == ${vocab_size}"
        lang_dir=data/lang_bpe_${vocab_size}
        out_dir=data/lm_training_bpe_${vocab_size}_userlibri
        mkdir -p $out_dir

        python3 ./utils/prepare_userlibri_lm_data.py --out-dir $out_dir
        ids=$(cat $out_dir/userlibri_ids.txt)


        for id in ${ids[@]}; do
            python3 ./local/prepare_lm_training_data.py \
                --bpe-model $lang_dir/bpe.model \
                --lm-data $out_dir/userlibri-lm-norm-$id.txt \
                --lm-archive $out_dir/lm_data_$id.pt
        done
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "Stage 4: Generate LM validation data"

  for vocab_size in ${vocab_sizes[@]}; do
    echo "Processing vocab_size == ${vocab_size}"
    out_dir=data/lm_training_bpe_${vocab_size}_userlibri
    mkdir -p $out_dir

    if [ ! -f $out_dir/valid.txt ]; then
      files=$(
        find "$dl_dir/LibriSpeech/dev-clean" -name "*.trans.txt"
        find "$dl_dir/LibriSpeech/dev-other" -name "*.trans.txt"
      )
      for f in ${files[@]}; do
        cat $f | cut -d " " -f 2-
      done > $out_dir/valid.txt
    fi

    lang_dir=data/lang_bpe_${vocab_size}
    ./local/prepare_lm_training_data.py \
      --bpe-model $lang_dir/bpe.model \
      --lm-data $out_dir/valid.txt \
      --lm-archive $out_dir/lm_data-valid.pt
  done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  echo "Stage 5: Generate LM test data"

  for vocab_size in ${vocab_sizes[@]}; do
    echo "Processing vocab_size == ${vocab_size}"
    out_dir=data/lm_training_bpe_${vocab_size}_userlibri
    mkdir -p $out_dir

    if [ ! -f $out_dir/test.txt ]; then
      files=$(
        find "$dl_dir/LibriSpeech/test-clean" -name "*.trans.txt"
        find "$dl_dir/LibriSpeech/test-other" -name "*.trans.txt"
      )
      for f in ${files[@]}; do
        cat $f | cut -d " " -f 2-
      done > $out_dir/test.txt
    fi

    lang_dir=data/lang_bpe_${vocab_size}
    ./local/prepare_lm_training_data.py \
      --bpe-model $lang_dir/bpe.model \
      --lm-data $out_dir/test.txt \
      --lm-archive $out_dir/lm_data-test.pt
  done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  echo "Stage 6: Sort LM training data"
  # Sort LM training data by sentence length in descending order
  # for ease of training.
  #
  # Sentence length equals to the number of BPE tokens
  # in a sentence.

  for vocab_size in ${vocab_sizes[@]}; do
    out_dir=data/lm_training_bpe_${vocab_size}_userlibri
    mkdir -p $out_dir
    ids=$(cat $out_dir/userlibri_ids.txt)
    for id in ${ids[@]}; do
        ./local/sort_lm_training_data.py \
        --in-lm-data $out_dir/lm_data_$id.pt \
        --out-lm-data $out_dir/sorted_lm_data_$id.pt \
        --out-statistics $out_dir/statistics_$id.txt
    done

    ./local/sort_lm_training_data.py \
      --in-lm-data $out_dir/lm_data-valid.pt \
      --out-lm-data $out_dir/sorted_lm_data-valid.pt \
      --out-statistics $out_dir/statistics-valid.txt

    ./local/sort_lm_training_data.py \
      --in-lm-data $out_dir/lm_data-test.pt \
      --out-lm-data $out_dir/sorted_lm_data-test.pt \
      --out-statistics $out_dir/statistics-test.txt
  done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  echo "Stage 7: test baseline RNN LM"

  ids=$(cat data/userlibri_test_id.txt)
  result_path="/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/result_baseline.txt"
  if [ -f $result_path ]; then rm $result_path; fi
  for id in ${ids[@]}; do
    echo $id
    ./pruned_transducer_stateless5/decode_userlibri.py \
      --epoch 30 \
      --avg 15 \
      --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
      --max-duration 400 \
      --decoding-method modified_beam_search_lm_shallow_fusion \
      --beam-size 4 \
      --num-encoder-layer 12 \
      --lm-type rnn \
      --lm-scale 0.15 \
      --lm-exp-dir rnn_lm/exp \
      --lm-epoch 30 \
      --lm-avg 1 \
      --rnn-lm-num-layers 3 \
      --use-shallow-fusion 1 \
      --rnn-lm-tie-weights 1 \
      --test-id $id || echo "ERROR OCCURS"
  done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  echo "Stage 8: train and decode using each book id"

  ids=$(cat data/userlibri_test_id.txt)
  result_path="/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/result_last_layer.txt"
  if [ -f $result_path ]; then rm $result_path; fi
  for id in ${ids[@]}; do
    echo $id
    num=$(echo $id | grep -o -E "[0-9]+")
    ./rnn_lm/train_plm.py \
      --start-epoch 31 \
      --world-size 4 \
      --num-epochs 35 \
      --use-fp16 0 \
      --embedding-dim 2048 \
      --hidden-dim 2048 \
      --num-layers 3 \
      --batch-size 100 \
      --exp-dir rnn_lm/exp \
      --train-n-layer 3 \
      --lm-data data/lm_training_bpe_500_userlibri/sorted_lm_data_$num.pt || echo "ERROR OCCURs"
    
    ./pruned_transducer_stateless5/decode_userlibri.py \
      --epoch 30 \
      --avg 15 \
      --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
      --max-duration 400 \
      --decoding-method modified_beam_search_lm_shallow_fusion \
      --beam-size 4 \
      --num-encoder-layer 12 \
      --lm-type rnn \
      --lm-scale 0.15 \
      --lm-exp-dir rnn_lm/exp \
      --lm-epoch 34 \
      --lm-avg 1 \
      --rnn-lm-num-layers 3 \
      --use-shallow-fusion 1 \
      --rnn-lm-tie-weights 1 \
      --test-id $id \
      --result-path $result_path  || echo "ERROR OCCURS"
  done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  echo "Stage 8: train and decode using each book id"

  ids=$(cat data/userlibri_test_id.txt)
  result_path="/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/result_surplus_random_init_only.txt"
  if [ -f $result_path ]; then rm $result_path; fi
  for id in ${ids[@]}; do
    echo $id
    num=$(echo $id | grep -o -E "[0-9]+")
    ./rnn_lm/train_plm.py \
      --start-epoch 31 \
      --world-size 4 \
      --num-epochs 35 \
      --use-fp16 0 \
      --embedding-dim 2048 \
      --hidden-dim 2048 \
      --num-layers 3 \
      --batch-size 100 \
      --exp-dir rnn_lm/exp \
      --surplus-layer true \
      --train-n-layer 4 \
      --lm-data data/lm_training_bpe_500_userlibri/sorted_lm_data_$num.pt || echo "ERROR OCCURs"
    ./pruned_transducer_stateless5/decode_userlibri.py \
      --epoch 30 \
      --avg 15 \
      --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
      --max-duration 400 \
      --decoding-method modified_beam_search_lm_shallow_fusion \
      --beam-size 4 \
      --num-encoder-layer 12 \
      --lm-type rnn \
      --lm-scale 0.15 \
      --lm-exp-dir rnn_lm/exp \
      --lm-epoch 34 \
      --lm-avg 1 \
      --rnn-lm-num-layers 3 \
      --use-shallow-fusion 1 \
      --rnn-lm-tie-weights 1 \
      --surplus-layer true \
      --test-id $id \
      --result-path $result_path  || echo "ERROR OCCURS"
  done
fi
if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  echo "Stage 8: train and decode using each book id"

  ids=$(cat data/userlibri_test_id.txt)
  result_path="/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/result_surplus_copy_last_only.txt"
  if [ -f $result_path ]; then rm $result_path; fi
  for id in ${ids[@]}; do
    echo $id
    num=$(echo $id | grep -o -E "[0-9]+")
    ./rnn_lm/train_plm.py \
      --start-epoch 31 \
      --world-size 4 \
      --num-epochs 35 \
      --use-fp16 0 \
      --embedding-dim 2048 \
      --hidden-dim 2048 \
      --num-layers 3 \
      --batch-size 100 \
      --exp-dir rnn_lm/exp \
      --train-n-layer 4 \
      --surplus-layer true \
      --copy-last-layer true \
      --lm-data data/lm_training_bpe_500_userlibri/sorted_lm_data_$num.pt || echo "ERROR OCCURs"
    ./pruned_transducer_stateless5/decode_userlibri.py \
      --epoch 30 \
      --avg 15 \
      --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
      --max-duration 400 \
      --decoding-method modified_beam_search_lm_shallow_fusion \
      --beam-size 4 \
      --num-encoder-layer 12 \
      --lm-type rnn \
      --lm-scale 0.15 \
      --lm-exp-dir rnn_lm/exp \
      --lm-epoch 34 \
      --lm-avg 1 \
      --rnn-lm-num-layers 3 \
      --use-shallow-fusion 1 \
      --rnn-lm-tie-weights 1 \
      --surplus-layer true \
      --test-id $id \
      --result-path $result_path  || echo "ERROR OCCURS"
  done
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  ids=(test-other_10136
  test-other_1184
  test-other_1399
  test-clean_15263
  test-clean_19215
  test-clean_2300
  test-clean_2681
  test-clean_2981
  test-clean_3178
  test-other_4276
  test-clean_507
  test-clean_732
  test-clean_820
  )
  lens=(38498 22062 18916 21088 13542 22202 8872 54306 8455 10461 9093 22980 11054)
  result_path="/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/result_per_1k.txt"
  if [ -f $result_path ]; then rm $result_path; fi
  for i in ${!ids[@]}; do
    len=${lens[i]}
    id=${ids[i]}
    echo $id
    num=$(echo $id | grep -o -E "[0-9]+")
    k=1000
    while [ $k -le $len ]
    do
      echo $k
      head -n $k data/lm_training_bpe_500_userlibri/userlibri-lm-norm-$num.txt > data/temp.txt
      python3 ./local/prepare_segmented_lm_training_data.py \
          --bpe-model data/lang_bpe_500/bpe.model \
          --lm-data data/temp.txt \
          --lm-archive data/temp.pt
      
      ./rnn_lm/train_plm.py \
        --start-epoch 31 \
        --world-size 4 \
        --num-epochs 35 \
        --use-fp16 0 \
        --embedding-dim 2048 \
        --hidden-dim 2048 \
        --num-layers 3 \
        --batch-size 100 \
        --exp-dir rnn_lm/exp \
        --train-n-layer 3 \
        --lm-data data/temp.pt || echo "ERROR OCCURs"

      ./pruned_transducer_stateless5/decode_userlibri.py \
        --epoch 30 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
        --max-duration 400 \
        --decoding-method modified_beam_search_lm_shallow_fusion \
        --beam-size 4 \
        --num-encoder-layer 12 \
        --lm-type rnn \
        --lm-scale 0.15 \
        --lm-exp-dir rnn_lm/exp \
        --lm-epoch 34 \
        --lm-avg 1 \
        --rnn-lm-num-layers 3 \
        --use-shallow-fusion 1 \
        --rnn-lm-tie-weights 1 \
        --test-id $id \
        --result-path $result_path  || echo "ERROR OCCURS"
      k=$((k+1000))
    done
  done
  rm data/temp.txt
  rm data/temp.pt
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
  echo "Stage 10: adapter training"

  ids=$(cat data/userlibri_test_id.txt)
  result_path="/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/result_adapter_16.txt"
  if [ -f $result_path ]; then rm $result_path; fi
  for id in ${ids[@]}; do
    echo $id
    num=$(echo $id | grep -o -E "[0-9]+")
    ./rnn_lm/train_plm.py \
      --start-epoch 31 \
      --world-size 4 \
      --num-epochs 35 \
      --use-fp16 0 \
      --embedding-dim 2048 \
      --hidden-dim 2048 \
      --num-layers 3 \
      --batch-size 100 \
      --exp-dir rnn_lm/exp \
      --train-n-layer 4 \
      --adapter true \
      --lm-data data/lm_training_bpe_500_userlibri/sorted_lm_data_$num.pt || echo "ERROR OCCURs"
    
    ./pruned_transducer_stateless5/decode_userlibri.py \
      --epoch 30 \
      --avg 15 \
      --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
      --max-duration 400 \
      --decoding-method modified_beam_search_lm_shallow_fusion \
      --beam-size 4 \
      --num-encoder-layer 12 \
      --lm-type rnn \
      --lm-scale 0.15 \
      --lm-exp-dir rnn_lm/exp \
      --lm-epoch 34 \
      --lm-avg 1 \
      --rnn-lm-num-layers 3 \
      --use-shallow-fusion 1 \
      --rnn-lm-tie-weights 1 \
      --adapter true \
      --test-id $id \
      --result-path $result_path  || echo "ERROR OCCURS"
  done
fi

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
  echo "Stage 11: train and weight average"

  ids=$(cat data/userlibri_test_id.txt)
  result_path="/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/result_last_layer.txt"
  if [ -f $result_path ]; then rm $result_path; fi
  if [ -f data/error_id.txt ]; then rm data/error_id.txt; fi
  for id in ${ids[@]}; do

    num=$(echo $id | grep -o -E "[0-9]+")
    if [ ! -d rnn_lm/exp_$id ]; then mkdir -p rnn_lm/exp_$id; fi
    cp rnn_lm/exp/epoch-30.pt rnn_lm/exp_$id/epoch-30.pt

    ./rnn_lm/train_plm.py \
      --start-epoch 31 \
      --world-size 4 \
      --num-epochs 35 \
      --use-fp16 0 \
      --embedding-dim 2048 \
      --hidden-dim 2048 \
      --num-layers 3 \
      --batch-size 100 \
      --exp-dir rnn_lm/exp_$id \
      --save-last-epoch true \
      --lm-data data/lm_training_bpe_500_userlibri/sorted_lm_data_$num.pt || (echo "ERROR OCCURs" && echo $id >> data/error_id.txt)
    if [ -f rnn_lm/exp_$id/epoch-30.pt ]; then rm rnn_lm/exp_$id/epoch-30.pt; fi
    
    ./pruned_transducer_stateless5/decode_userlibri.py \
      --epoch 30 \
      --avg 15 \
      --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
      --max-duration 400 \
      --decoding-method modified_beam_search_lm_shallow_fusion \
      --beam-size 4 \
      --num-encoder-layer 12 \
      --lm-type rnn \
      --lm-scale 0.15 \
      --lm-exp-dir rnn_lm/exp_$id \
      --lm-epoch 34 \
      --lm-avg 1 \
      --rnn-lm-num-layers 3 \
      --use-shallow-fusion 1 \
      --rnn-lm-tie-weights 1 \
      --test-id $id \
      --result-path $result_path  || (echo "ERROR OCCURS" && echo $id >> data/error_id.txt && rm -r rnn_lm/exp_$id)
  done

  ./utils/average_parameters.py \
    --data-id "${ids}" \
    --start-epoch 35 \
    --exp-dir rnn_lm/exp_averaged
fi

if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
  echo "Stage 12: average_model decoding"

  ids=$(cat data/userlibri_test_id.txt)
  result_path="/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/result_average_whole-layers.txt"
  if [ -f $result_path ]; then rm $result_path; fi
  for id in ${ids[@]}; do
    echo $id
    num=$(echo $id | grep -o -E "[0-9]+")
    
    # ./rnn_lm/train_plm.py \
    #   --start-epoch 35 \
    #   --world-size 4 \
    #   --num-epochs 45 \
    #   --use-fp16 0 \
    #   --embedding-dim 2048 \
    #   --hidden-dim 2048 \
    #   --num-layers 3 \
    #   --batch-size 100 \
    #   --exp-dir rnn_lm/exp_averaged \
    #   --save-last-epoch true \
    #   --lm-data data/lm_training_bpe_500_userlibri/sorted_lm_data_$num.pt

    ./pruned_transducer_stateless5/decode_userlibri.py \
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
      --lm-epoch 34 \
      --lm-avg 1 \
      --rnn-lm-num-layers 3 \
      --use-shallow-fusion 1 \
      --rnn-lm-tie-weights 1 \
      --test-id $id \
      --result-path $result_path
  done
fi

if [ $stage -le 13 ] && [ $stop_stage -ge 13 ]; then
  echo "Stage 13: train and decode using each book id for 45 epochs"
  filename="data/book_to_data.txt"
  for epoch in 75 80; do
    result_path="/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_vanilla_$epoch.txt"
    if [ -f $result_path ]; then rm $result_path; fi
    while read line; do
      book_id=$(echo "$line" | cut -f1)
      data_ids=$(echo "$line" | cut -f2-)
      cp rnn_lm/exp/epoch-30.pt rnn_lm/exp_book_$book_id/epoch-30.pt
      if [ $book_id -eq 2002 ]
      then
        batch_size=50
      else
        batch_size=100
      fi
      ./rnn_lm/train_plm.py \
        --start-epoch $((epoch-4)) \
        --world-size 3 \
        --num-epochs $((epoch+1)) \
        --use-fp16 0 \
        --embedding-dim 2048 \
        --hidden-dim 2048 \
        --num-layers 3 \
        --batch-size $batch_size \
        --exp-dir rnn_lm/exp_book_$book_id \
        --save-last-epoch true \
        --lm-data-path data/lm_training_bpe_500_userlibri \
        --lm-data-name "$book_id"
      
      ./pruned_transducer_stateless5/decode_userlibri.py \
        --epoch 30 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
        --max-duration 400 \
        --decoding-method modified_beam_search_lm_shallow_fusion \
        --beam-size 4 \
        --num-encoder-layer 12 \
        --lm-type rnn \
        --lm-scale 0.15 \
        --lm-exp-dir rnn_lm/exp_book_$book_id \
        --lm-epoch $epoch \
        --lm-avg 1 \
        --rnn-lm-num-layers 3 \
        --use-shallow-fusion 1 \
        --rnn-lm-tie-weights 1 \
        --test-id "$data_ids" \
        --book-id "$book_id" \
        --result-path $result_path
    done < "$filename"
  done
fi

if [ $stage -le 14 ] && [ $stop_stage -ge 14 ]; then
  echo "Stage 14: average top 20 model"

  ids=$(cat data/userlibri_test_id.txt)
  result_path="/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/plm_dev_results.txt"
  if [ -f $result_path ]; then rm $result_path; fi
  for id in ${ids[@]}; do

    num=$(echo $id | grep -o -E "[0-9]+")
    if [ ! -d rnn_lm/exp_$id ]; then mkdir -p rnn_lm/exp_$id; fi
    cp rnn_lm/exp/epoch-30.pt rnn_lm/exp_$id/epoch-30.pt

    ./rnn_lm/train_plm.py \
      --start-epoch 31 \
      --world-size 4 \
      --num-epochs 35 \
      --use-fp16 0 \
      --embedding-dim 2048 \
      --hidden-dim 2048 \
      --num-layers 3 \
      --batch-size 100 \
      --exp-dir rnn_lm/exp_$id \
      --save-last-epoch true \
      --lm-data data/lm_training_bpe_500_userlibri/sorted_lm_data_$num.pt || (echo "ERROR OCCURs" && echo $id >> data/error_id.txt)
    if [ -f rnn_lm/exp_$id/epoch-30.pt ]; then rm rnn_lm/exp_$id/epoch-30.pt; fi
    
    ./pruned_transducer_stateless5/decode_userlibri.py \
      --epoch 30 \
      --avg 15 \
      --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
      --max-duration 400 \
      --decoding-method modified_beam_search_lm_shallow_fusion \
      --beam-size 4 \
      --num-encoder-layer 12 \
      --lm-type rnn \
      --lm-scale 0.15 \
      --lm-exp-dir rnn_lm/exp_$id \
      --lm-epoch 34 \
      --lm-avg 1 \
      --rnn-lm-num-layers 3 \
      --use-shallow-fusion 1 \
      --rnn-lm-tie-weights 1 \
      --test-id $id \
      --result-path $result_path  || (echo "ERROR OCCURS" && echo $id >> data/error_id.txt && rm -r rnn_lm/exp_$id)
  done

  # ./utils/average_parameters.py \
  #   --data-id "${result_path}" \
  #   --start-epoch 35 \
  #   --topk 20
  #   --exp-dir rnn_lm/exp_averaged
fi

if [ $stage -le 15 ] && [ $stop_stage -ge 15 ]; then
  echo "Stage 15: average top k model and decoding"

  ids=$(cat data/userlibri_test_id.txt)
  for i in {5..30}; do
    echo $i
    result_path2=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/plm_average_topk${i}.txt
    if [ -f $result_path2 ]; then rm $result_path2; fi
    lm_list=pruned_transducer_stateless5/plm_dev_results.txt
    ./utils/average_parameters_topk.py \
      --data-id "${ids}" \
      --lm-list ${lm_list} \
      --start-epoch 35 \
      --topk $i \
      --exp-dir rnn_lm/exp_averaged
    for id in ${ids[@]}; do
      ./pruned_transducer_stateless5/decode_userlibri.py \
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
        --lm-epoch 34 \
        --lm-avg 1 \
        --rnn-lm-num-layers 3 \
        --use-shallow-fusion 1 \
        --rnn-lm-tie-weights 1 \
        --test-id $id \
        --result-path $result_path2
    done

    result_path=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/plm_average_topk_userlibri_${i}.txt
    if [ -f $result_path ]; then rm $result_path; fi

    for id in ${ids[@]}; do
      num=$(echo $id | grep -o -E "[0-9]+")
      ./rnn_lm/train_plm.py \
        --start-epoch 35 \
        --world-size 4 \
        --num-epochs 40 \
        --use-fp16 0 \
        --embedding-dim 2048 \
        --hidden-dim 2048 \
        --num-layers 3 \
        --batch-size 100 \
        --exp-dir rnn_lm/exp_averaged \
        --save-last-epoch true \
        --lm-data data/lm_training_bpe_500_userlibri/sorted_lm_data_$num.pt    

      ./pruned_transducer_stateless5/decode_userlibri.py \
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
        --lm-epoch 39 \
        --lm-avg 1 \
        --rnn-lm-num-layers 3 \
        --use-shallow-fusion 1 \
        --rnn-lm-tie-weights 1 \
        --test-id $id \
        --result-path $result_path
    done
  done
fi

if [ $stage -le 16 ] && [ $stop_stage -ge 16 ]; then
  echo "Stage 16: train per spk id"

  filename="data/id_to_books.txt"
  result_path="pruned_transducer_stateless5/results_per_spkid.txt"
  result_path2="pruned_transducer_stateless5/results_per_spkid_baseline.txt"

  if [ -f $result_path ]; then rm $result_path; fi
  if [ -f $result_path2 ]; then rm $result_path2; fi


  while read line; do
    # echo $line
    spk_id=$(echo "$line" | cut -f1)
    book_ids=$(echo "$line" | cut -f2-)
    num=$(echo $book_ids | grep -o -E "[0-9]+")
    
    if [ ! -d rnn_lm/exp_$spk_id ]; then mkdir -p rnn_lm/exp_$spk_id; fi
    cp rnn_lm/exp/epoch-30.pt rnn_lm/exp_"$spk_id"
    ./rnn_lm/train_plm_perid.py \
      --start-epoch 31 \
      --world-size 4 \
      --num-epochs 40 \
      --use-fp16 0 \
      --embedding-dim 2048 \
      --hidden-dim 2048 \
      --num-layers 3 \
      --batch-size 100 \
      --exp-dir rnn_lm/exp_$spk_id \
      --save-last-epoch true \
      --lm-data-path data/lm_training_bpe_500_userlibri \
      --lm-data-name "$spk_id"
      #--lm-data-path data/lm_training_bpe_500_userlibri/sorted_lm_data_$num.pt \
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
      --lm-epoch 39 \
      --lm-avg 1 \
      --rnn-lm-num-layers 3 \
      --use-shallow-fusion 1 \
      --rnn-lm-tie-weights 1 \
      --test-id "$book_ids" \
      --spk-id "$spk_id" \
      --result-path $result_path
  done < "$filename"
  
  while read line; do
    # echo $line
    spk_id=$(echo "$line" | cut -f1)
    book_ids=$(echo "$line" | cut -f2-)
    echo $spk_id
    echo $book_ids
    num=$(echo $book_ids | grep -o -E "[0-9]+")
    echo $num

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
      --lm-exp-dir rnn_lm/exp \
      --lm-epoch 30 \
      --lm-avg 1 \
      --rnn-lm-num-layers 3 \
      --use-shallow-fusion 1 \
      --rnn-lm-tie-weights 1 \
      --test-id "$book_ids" \
      --spk-id "$spk_id" \
      --result-path $result_path2
  done < "$filename"
fi

if [ $stage -le 17 ] && [ $stop_stage -ge 17 ]; then
  echo "Stage 17: train per spk id with MAML fed"

  filename="data/id_to_books.txt"

  for epoch in 35 40 45 50; do
    result_path=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_spkid_fed_"$epoch"_test.txt
    result_path2=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_spkid_fed_"$epoch"_avg_test.txt

    if [ -f $result_path ]; then rm $result_path; fi
    if [ -f $result_path2 ]; then rm $result_path2; fi
    while read line; do
      # echo $line
      spk_id=$(echo "$line" | cut -f1)
      book_ids=$(echo "$line" | cut -f2-)
      echo $spk_id
      echo $book_ids
      num=$(echo $book_ids | grep -o -E "[0-9]+")
      echo $num
      if [ ! -d rnn_lm/exp_$spk_id ]; then mkdir -p rnn_lm/exp_$spk_id; fi
      cp rnn_lm/exp_fed/epoch-$((epoch-5)).pt rnn_lm/exp_"$spk_id"/epoch-$((epoch-5)).pt

      ./rnn_lm/train_plm_fed_perid.py \
        --start-epoch $((epoch-4)) \
        --world-size 4 \
        --num-epochs $((epoch+1)) \
        --use-fp16 0 \
        --embedding-dim 2048 \
        --hidden-dim 2048 \
        --num-layers 3 \
        --batch-size 100 \
        --exp-dir rnn_lm/exp_$spk_id \
        --save-last-epoch true \
        --lm-data-path data/lm_training_bpe_500_userlibri \
        --lm-data-name "$spk_id" \
        --alpha 1e-4 \
        --beta 1e-3


        #--lm-data-path data/lm_training_bpe_500_userlibri/sorted_lm_data_$num.pt \
      if [ -f rnn_lm/exp_$spk_id/epoch-$((epoch-5)).pt ]; then rm rnn_lm/exp_$spk_id/epoch-$((epoch-5)).pt; fi
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
    lm_list=pruned_transducer_stateless5/results_per_spkid_fed_"$epoch".txt
    ./utils/average_parameters_fed.py \
      --topk 67 \
      --lm-list ${lm_list} \
      --start-epoch $((epoch+1)) \
      --exp-dir rnn_lm/exp_fed
    
    while read line; do
      # echo $line
      spk_id=$(echo "$line" | cut -f1)
      book_ids=$(echo "$line" | cut -f2-)
      echo $spk_id
      echo $book_ids
      num=$(echo $book_ids | grep -o -E "[0-9]+")
      echo $num

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
fi

if [ $stage -le 18 ] && [ $stop_stage -ge 18 ]; then
  echo "Stage 18: train per book id with MAML fed"
  mkdir -p rnn_lm/exp_fed_perbook
  if [ ! -f rnn_lm/exp_fed_perbook/epoch-30.pt ]; then cp rnn_lm/exp/epoch-30.pt rnn_lm/exp_fed_perbook; fi
  filename="data/book_to_data.txt"
  alpha=(1e-4)
  beta=(1e-1)

  for i in "${!alpha[@]}"; do
    for epoch in 35 40 45 50 55 60 65 70; do
      result_path=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_fed2_"$epoch"_${alpha[$i]}_${beta[$i]}.txt
      result_path2=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_fed2_"$epoch"_${alpha[$i]}_${beta[$i]}_avg.txt

      if [ -f $result_path ]; then rm $result_path; fi
      if [ -f $result_path2 ]; then rm $result_path2; fi
      echo ${alpha[$i]}_${beta[$i]}
      while read line; do
        book_id=$(echo "$line" | cut -f1)
        data_ids=$(echo "$line" | cut -f2-)
        # echo $line
        if [ ! -d rnn_lm/exp_book_$book_id ]; then mkdir -p rnn_lm/exp_book_$book_id; fi
        cp rnn_lm/exp_fed_perbook/epoch-$((epoch-5)).pt rnn_lm/exp_book_$book_id/epoch-$((epoch-5)).pt
        
        if [ $book_id -eq 2002 ]
        then
          batch_size=50
        else
          batch_size=100
        fi
        # echo $batch_size

        ./rnn_lm/train_plm_fed_perbook.py \
          --start-epoch $((epoch-4)) \
          --world-size 3 \
          --num-epochs $((epoch+1)) \
          --use-fp16 0 \
          --embedding-dim 2048 \
          --hidden-dim 2048 \
          --num-layers 3 \
          --batch-size $batch_size \
          --exp-dir rnn_lm/exp_book_$book_id \
          --save-last-epoch true \
          --lm-data-path data/lm_training_bpe_500_userlibri \
          --lm-data-name "$book_id" \
          --alpha ${alpha[$i]} \
          --beta ${beta[$i]}
        
        # if [ "$epoch" != 40 ]; then
        #   if [ -f rnn_lm/exp_$id/epoch-$((epoch-5)).pt ]; then rm rnn_lm/exp_$id/epoch-$((epoch-5)).pt; fi
        # fi

        if [ -f rnn_lm/exp_book_$book_id/epoch-30.pt ]; then rm rnn_lm/exp_book_$book_id/epoch-30.pt; fi
        ./pruned_transducer_stateless5/decode_userlibri.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_book_$book_id \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$data_ids" \
          --book-id "$book_id" \
          --result-path $result_path
          
      done < "$filename"

      lm_list=pruned_transducer_stateless5/results_per_book_fed2_"$epoch"_${alpha[$i]}_${beta[$i]}.txt
      ./utils/average_parameters_fed.py \
        --lm-list ${lm_list} \
        --start-epoch $((epoch+1)) \
        --exp-dir rnn_lm/exp_fed_perbook
      
      while read line; do
        book_id=$(echo "$line" | cut -f1)
        data_ids=$(echo "$line" | cut -f2-)
        ./pruned_transducer_stateless5/decode_userlibri.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_fed_perbook \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$data_ids" \
          --book-id "$book_id" \
          --result-path $result_path2
      done < "$filename"
    done

    echo "Stage 18-1: overfitting stage"

    for epoch in 75 80 85 90; do
      
      result_path=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_fed2_overfitting_"$epoch".txt

      if [ -f $result_path ]; then rm $result_path; fi
      while read line; do
        book_id=$(echo "$line" | cut -f1)
        data_ids=$(echo "$line" | cut -f2-)
        cp rnn_lm/exp_fed_perbook/epoch-45.pt rnn_lm/exp_book_$book_id/epoch-45.pt
        if [ $book_id -eq 2002 ]
        then
          batch_size=50
        else
          batch_size=100
        fi
        ./rnn_lm/train_plm.py \
          --start-epoch $((epoch-4)) \
          --world-size 3 \
          --num-epochs $((epoch+1)) \
          --use-fp16 0 \
          --embedding-dim 2048 \
          --hidden-dim 2048 \
          --num-layers 3 \
          --batch-size $batch_size \
          --exp-dir rnn_lm/exp_book_$book_id \
          --save-last-epoch true \
          --lm-data-path data/lm_training_bpe_500_userlibri \
          --lm-data-name "$book_id"
        
        ./pruned_transducer_stateless5/decode_userlibri.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_book_$book_id \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$data_ids" \
          --book-id "$book_id" \
          --result-path $result_path
      done < "$filename"
    done
  done

  # while read line; do
  #   book_id=$(echo "$line" | cut -f1)
  #   data_ids=$(echo "$line" | cut -f2-)
  #   ./pruned_transducer_stateless5/decode_userlibri.py \
  #     --epoch 30 \
  #     --avg 15 \
  #     --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
  #     --max-duration 400 \
  #     --decoding-method modified_beam_search_lm_shallow_fusion \
  #     --beam-size 4 \
  #     --num-encoder-layer 12 \
  #     --lm-type rnn \
  #     --lm-scale 0.15 \
  #     --lm-exp-dir rnn_lm/exp \
  #     --lm-epoch 30 \
  #     --lm-avg 1 \
  #     --rnn-lm-num-layers 3 \
  #     --use-shallow-fusion 1 \
  #     --rnn-lm-tie-weights 1 \
  #     --test-id "$data_ids" \
  #     --book-id "$book_id" \
  #     --result-path pruned_transducer_stateless5/result_per_book_baseline.txt
  # done < "$filename"
fi

if [ $stage -le 19 ] && [ $stop_stage -ge 19 ]; then
  echo "Stage 19: mix baseline and fed lm"
  filename="data/book_to_data.txt"
  for alpha in $(seq 0 0.1 1); do
    echo $alpha
    filename="data/book_to_data.txt"
    ./utils/mix_two_model.py \
      --model-paths "rnn_lm/exp_fed_perbook/epoch-45.pt rnn_lm/exp/epoch-30.pt" \
      --start-epoch 45 \
      --exp-dir rnn_lm/exp_averaged \
      --alpha ${alpha}
    result_path3=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_convex_${alpha}.txt
    if [ -f $result_path3 ]; then rm $result_path3; fi
    while read line; do
      book_id=$(echo "$line" | cut -f1)
      data_ids=$(echo "$line" | cut -f2-)
      

      ./pruned_transducer_stateless5/decode_userlibri.py \
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
        --lm-epoch 45 \
        --lm-avg 1 \
        --rnn-lm-num-layers 3 \
        --use-shallow-fusion 1 \
        --rnn-lm-tie-weights 1 \
        --test-id "$data_ids" \
        --book-id "$book_id" \
        --result-path $result_path3
    done < "$filename"
  done
fi


if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]; then
  echo "Stage 20: train and average"
  mkdir -p rnn_lm/exp_averaged
  if [ ! -f rnn_lm/exp_averaged/epoch-30.pt ]; then cp rnn_lm/exp/epoch-30.pt rnn_lm/exp_averaged; fi
  filename="data/book_to_data.txt"
  alpha=(1e-3 1e-3 1e-3 1e-4 1e-4 1e-2 1e-2 1e-2)
  beta=(1e-2 1e-3 1e-4 1e-2 1e-3 1e-2 1e-3 1e-4)

  for epoch in 35 40 45; do
    result_path=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_"$epoch".txt
    result_path2=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_"$epoch"_avg.txt

    if [ -f $result_path ]; then rm $result_path; fi
    if [ -f $result_path2 ]; then rm $result_path2; fi

    while read line; do
      book_id=$(echo "$line" | cut -f1)
      data_ids=$(echo "$line" | cut -f2-)
      # echo $line
      if [ ! -d rnn_lm/exp_book_$book_id ]; then mkdir -p rnn_lm/exp_book_$book_id; fi
      cp rnn_lm/exp_averaged/epoch-$((epoch-5)).pt rnn_lm/exp_book_$book_id/epoch-$((epoch-5)).pt
      
      if [ $book_id -eq 2002 ]
      then
        batch_size=50
      else
        batch_size=100
      fi
      # echo $batch_size

      ./rnn_lm/train_plm.py \
        --start-epoch $((epoch-4)) \
        --world-size 3 \
        --num-epochs $((epoch+1)) \
        --use-fp16 0 \
        --embedding-dim 2048 \
        --hidden-dim 2048 \
        --num-layers 3 \
        --batch-size $batch_size \
        --exp-dir rnn_lm/exp_book_$book_id \
        --save-last-epoch true \
        --lm-data-path data/lm_training_bpe_500_userlibri \
        --lm-data-name "$book_id" \
      
      # if [ "$epoch" != 40 ]; then
      #   if [ -f rnn_lm/exp_$id/epoch-$((epoch-5)).pt ]; then rm rnn_lm/exp_$id/epoch-$((epoch-5)).pt; fi
      # fi
      if [ -f rnn_lm/exp_book_$book_id/epoch-30.pt ]; then rm rnn_lm/exp_book_$book_id/epoch-30.pt; fi
      ./pruned_transducer_stateless5/decode_userlibri.py \
        --epoch 30 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
        --max-duration 400 \
        --decoding-method modified_beam_search_lm_shallow_fusion \
        --beam-size 4 \
        --num-encoder-layer 12 \
        --lm-type rnn \
        --lm-scale 0.15 \
        --lm-exp-dir rnn_lm/exp_book_$book_id \
        --lm-epoch $epoch \
        --lm-avg 1 \
        --rnn-lm-num-layers 3 \
        --use-shallow-fusion 1 \
        --rnn-lm-tie-weights 1 \
        --test-id "$data_ids" \
        --book-id "$book_id" \
        --result-path $result_path 
    done < "$filename"

    lm_list=pruned_transducer_stateless5/results_per_book_"$epoch".txt
    ./utils/average_parameters_fed.py \
      --lm-list ${lm_list} \
      --start-epoch $((epoch+1)) \
      --exp-dir rnn_lm/exp_averaged
    
    while read line; do
      book_id=$(echo "$line" | cut -f1)
      data_ids=$(echo "$line" | cut -f2-)
      ./pruned_transducer_stateless5/decode_userlibri.py \
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
        --test-id "$data_ids" \
        --book-id "$book_id" \
        --result-path $result_path2
    done < "$filename"
  done
  echo "Stage 20-1: overfitting stage"

  for epoch in 50 55 60 65 70 75 80; do
    
    result_path=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_overfitting_"$epoch".txt

    if [ -f $result_path ]; then rm $result_path; fi
    while read line; do
      book_id=$(echo "$line" | cut -f1)
      data_ids=$(echo "$line" | cut -f2-)
      cp rnn_lm/exp_averaged/epoch-45.pt rnn_lm/exp_book_$book_id/epoch-45.pt
      if [ $book_id -eq 2002 ]
      then
        batch_size=50
      else
        batch_size=100
      fi
      ./rnn_lm/train_plm.py \
        --start-epoch $((epoch-4)) \
        --world-size 3 \
        --num-epochs $((epoch+1)) \
        --use-fp16 0 \
        --embedding-dim 2048 \
        --hidden-dim 2048 \
        --num-layers 3 \
        --batch-size $batch_size \
        --exp-dir rnn_lm/exp_book_$book_id \
        --save-last-epoch true \
        --lm-data-path data/lm_training_bpe_500_userlibri \
        --lm-data-name "$book_id"
      
      ./pruned_transducer_stateless5/decode_userlibri.py \
        --epoch 30 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
        --max-duration 400 \
        --decoding-method modified_beam_search_lm_shallow_fusion \
        --beam-size 4 \
        --num-encoder-layer 12 \
        --lm-type rnn \
        --lm-scale 0.15 \
        --lm-exp-dir rnn_lm/exp_book_$book_id \
        --lm-epoch $epoch \
        --lm-avg 1 \
        --rnn-lm-num-layers 3 \
        --use-shallow-fusion 1 \
        --rnn-lm-tie-weights 1 \
        --test-id "$data_ids" \
        --book-id "$book_id" \
        --result-path $result_path
    done < "$filename"
  done
fi

if [ $stage -le 21 ] && [ $stop_stage -ge 21 ]; then
  echo "Stage 21: FED + adaptive weight average wrt valid WER"
  mkdir -p rnn_lm/exp_fed_perbook
  if [ ! -f rnn_lm/exp_fed_perbook/epoch-30.pt ]; then cp rnn_lm/exp/epoch-30.pt rnn_lm/exp_fed_perbook; fi
  filename="data/book_to_data.txt"
  alpha=(1e-4)
  beta=(1e-1)

  for i in "${!alpha[@]}"; do
    for epoch in 35 40 45; do
      result_path=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_fed_wer2_"$epoch"_${alpha[$i]}_${beta[$i]}.txt
      result_path2=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_fed_wer2_"$epoch"_${alpha[$i]}_${beta[$i]}_valid.txt
      result_path3=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_fed_wer2_"$epoch"_${alpha[$i]}_${beta[$i]}_avg.txt

      if [ -f $result_path ]; then rm $result_path; fi
      if [ -f $result_path2 ]; then rm $result_path2; fi
      if [ -f $result_path3 ]; then rm $result_path3; fi
      echo ${alpha[$i]}_${beta[$i]}
      while read line; do
        book_id=$(echo "$line" | cut -f1)
        data_ids=$(echo "$line" | cut -f2-)
        # echo $line
        if [ ! -d rnn_lm/exp_book_$book_id ]; then mkdir -p rnn_lm/exp_book_$book_id; fi
        cp rnn_lm/exp_fed_perbook/epoch-$((epoch-5)).pt rnn_lm/exp_book_$book_id/epoch-$((epoch-5)).pt
        
        if [ $book_id -eq 2002 ]
        then
          batch_size=50
        else
          batch_size=100
        fi
        # echo $batch_size

        ./rnn_lm/train_plm_fed_perbook.py \
          --start-epoch $((epoch-4)) \
          --world-size 3 \
          --num-epochs $((epoch+1)) \
          --use-fp16 0 \
          --embedding-dim 2048 \
          --hidden-dim 2048 \
          --num-layers 3 \
          --batch-size $batch_size \
          --exp-dir rnn_lm/exp_book_$book_id \
          --save-last-epoch true \
          --lm-data-path data/lm_training_bpe_500_userlibri \
          --lm-data-name "$book_id" \
          --alpha ${alpha[$i]} \
          --beta ${beta[$i]}
        
        # if [ "$epoch" != 40 ]; then
        #   if [ -f rnn_lm/exp_$id/epoch-$((epoch-5)).pt ]; then rm rnn_lm/exp_$id/epoch-$((epoch-5)).pt; fi
        # fi

        if [ -f rnn_lm/exp_book_$book_id/epoch-30.pt ]; then rm rnn_lm/exp_book_$book_id/epoch-30.pt; fi
        ./pruned_transducer_stateless5/decode_userlibri.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_book_$book_id \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$data_ids" \
          --book-id "$book_id" \
          --result-path $result_path
          
      done < "$filename"

      while read line; do
        book_id=$(echo "$line" | cut -f1)
        data_ids=$(echo "$line" | cut -f2-)
        ./pruned_transducer_stateless5/decode_userlibri_valid.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_book_$book_id \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$data_ids" \
          --book-id "$book_id" \
          --result-path $result_path2
      done < "$filename"

      lm_list=pruned_transducer_stateless5/results_per_book_fed_wer2_"$epoch"_${alpha[$i]}_${beta[$i]}_valid.txt
      ./utils/average_parameters_fed_WER.py \
        --lm-list ${lm_list} \
        --start-epoch $((epoch+1)) \
        --exp-dir rnn_lm/exp_fed_perbook
      
      while read line; do
        book_id=$(echo "$line" | cut -f1)
        data_ids=$(echo "$line" | cut -f2-)
        ./pruned_transducer_stateless5/decode_userlibri.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_fed_perbook \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$data_ids" \
          --book-id "$book_id" \
          --result-path $result_path3
      done < "$filename"
    done

    echo "Stage 18-1: overfitting stage"

    for epoch in 50 55 60 65 70 75 80; do
      
      result_path=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_fed_wer2_overfitting_"$epoch".txt

      if [ -f $result_path ]; then rm $result_path; fi
      while read line; do
        book_id=$(echo "$line" | cut -f1)
        data_ids=$(echo "$line" | cut -f2-)
        cp rnn_lm/exp_fed_perbook/epoch-45.pt rnn_lm/exp_book_$book_id/epoch-45.pt
        if [ $book_id -eq 2002 ]
        then
          batch_size=50
        else
          batch_size=100
        fi
        ./rnn_lm/train_plm.py \
          --start-epoch $((epoch-4)) \
          --world-size 3 \
          --num-epochs $((epoch+1)) \
          --use-fp16 0 \
          --embedding-dim 2048 \
          --hidden-dim 2048 \
          --num-layers 3 \
          --batch-size $batch_size \
          --exp-dir rnn_lm/exp_book_$book_id \
          --save-last-epoch true \
          --lm-data-path data/lm_training_bpe_500_userlibri \
          --lm-data-name "$book_id"
        
        ./pruned_transducer_stateless5/decode_userlibri.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_book_$book_id \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$data_ids" \
          --book-id "$book_id" \
          --result-path $result_path
      done < "$filename"
    done
  done
fi

if [ $stage -le 22 ] && [ $stop_stage -ge 22 ]; then
  echo "Stage 22: FED3: CV + FED + model selection"
  mkdir -p rnn_lm/exp_fed_perbook
  if [ ! -f rnn_lm/exp_fed_perbook/epoch-30.pt ]; then cp rnn_lm/exp/epoch-30.pt rnn_lm/exp_fed_perbook; fi
  filename="data/book_to_data.txt"
  alpha=(1e-4)
  beta=(1e-1)

  for i in "${!alpha[@]}"; do
    for epoch in 35 40 45; do
      result_path=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_fed3_"$epoch"_${alpha[$i]}_${beta[$i]}.txt
      result_path2=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_fed3_"$epoch"_${alpha[$i]}_${beta[$i]}_valid.txt
      result_path3=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_fed3_"$epoch"_${alpha[$i]}_${beta[$i]}_avg.txt

      if [ -f $result_path ]; then rm $result_path; fi
      if [ -f $result_path2 ]; then rm $result_path2; fi
      if [ -f $result_path3 ]; then rm $result_path3; fi
      echo ${alpha[$i]}_${beta[$i]}
      while read line; do
        book_id=$(echo "$line" | cut -f1)
        data_ids=$(echo "$line" | cut -f2-)
        # echo $line
        if [ ! -d rnn_lm/exp_book_$book_id ]; then mkdir -p rnn_lm/exp_book_$book_id; fi
        cp rnn_lm/exp_fed_perbook/epoch-$((epoch-5)).pt rnn_lm/exp_book_$book_id/epoch-$((epoch-5)).pt
        
        if [ $book_id -eq 2002 ]
        then
          batch_size=50
        else
          batch_size=100
        fi
        # echo $batch_size

        ./rnn_lm/train_plm_fed_perbook.py \
          --start-epoch $((epoch-4)) \
          --world-size 3 \
          --num-epochs $((epoch+1)) \
          --use-fp16 0 \
          --embedding-dim 2048 \
          --hidden-dim 2048 \
          --num-layers 3 \
          --batch-size $batch_size \
          --exp-dir rnn_lm/exp_book_$book_id \
          --save-last-epoch true \
          --lm-data-path data/lm_training_bpe_500_userlibri \
          --lm-data-name "$book_id" \
          --alpha ${alpha[$i]} \
          --beta ${beta[$i]}
        
        # if [ "$epoch" != 40 ]; then
        #   if [ -f rnn_lm/exp_$id/epoch-$((epoch-5)).pt ]; then rm rnn_lm/exp_$id/epoch-$((epoch-5)).pt; fi
        # fi

        if [ -f rnn_lm/exp_book_$book_id/epoch-30.pt ]; then rm rnn_lm/exp_book_$book_id/epoch-30.pt; fi
        ./pruned_transducer_stateless5/decode_userlibri.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_book_$book_id \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$data_ids" \
          --book-id "$book_id" \
          --result-path $result_path
          
      done < "$filename"

      while read line; do
        book_id=$(echo "$line" | cut -f1)
        data_ids=$(echo "$line" | cut -f2-)
        ./pruned_transducer_stateless5/decode_userlibri_valid.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_book_$book_id \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$data_ids" \
          --book-id "$book_id" \
          --result-path $result_path2
      done < "$filename"

      lm_list=pruned_transducer_stateless5/results_per_book_fed3_"$epoch"_${alpha[$i]}_${beta[$i]}_valid.txt
      ./utils/average_parameters_fed.py \
        --lm-list ${lm_list} \
        --start-epoch $((epoch+1)) \
        --exp-dir rnn_lm/exp_fed_perbook
      
      while read line; do
        book_id=$(echo "$line" | cut -f1)
        data_ids=$(echo "$line" | cut -f2-)
        ./pruned_transducer_stateless5/decode_userlibri.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_fed_perbook \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$data_ids" \
          --book-id "$book_id" \
          --result-path $result_path3
      done < "$filename"
    done

    echo "Stage 22-1: overfitting stage"

    for epoch in 50 55 60 65 70 75 80; do
      
      result_path=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_fed3_overfitting_"$epoch".txt

      if [ -f $result_path ]; then rm $result_path; fi
      while read line; do
        book_id=$(echo "$line" | cut -f1)
        data_ids=$(echo "$line" | cut -f2-)
        cp rnn_lm/exp_fed_perbook/epoch-45.pt rnn_lm/exp_book_$book_id/epoch-45.pt
        if [ $book_id -eq 2002 ]
        then
          batch_size=50
        else
          batch_size=100
        fi
        ./rnn_lm/train_plm.py \
          --start-epoch $((epoch-4)) \
          --world-size 3 \
          --num-epochs $((epoch+1)) \
          --use-fp16 0 \
          --embedding-dim 2048 \
          --hidden-dim 2048 \
          --num-layers 3 \
          --batch-size $batch_size \
          --exp-dir rnn_lm/exp_book_$book_id \
          --save-last-epoch true \
          --lm-data-path data/lm_training_bpe_500_userlibri \
          --lm-data-name "$book_id"
        
        ./pruned_transducer_stateless5/decode_userlibri.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_book_$book_id \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$data_ids" \
          --book-id "$book_id" \
          --result-path $result_path
      done < "$filename"
    done
  done
fi

if [ $stage -le 23 ] && [ $stop_stage -ge 23 ]; then
  echo "Stage 23: train per book id with MAML fed, finetuning for 10epoch"
  mkdir -p rnn_lm/exp_fed_perbook
  if [ ! -f rnn_lm/exp_fed_perbook/epoch-30.pt ]; then cp rnn_lm/exp/epoch-30.pt rnn_lm/exp_fed_perbook; fi
  filename="data/book_to_data.txt"
  alpha=(1e-4)
  beta=(1e-1)

  for i in "${!alpha[@]}"; do
    for epoch in 40 50 60; do
      result_path=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_fed4_"$epoch"_${alpha[$i]}_${beta[$i]}.txt
      result_path2=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_fed4_"$epoch"_${alpha[$i]}_${beta[$i]}_avg.txt

      if [ -f $result_path ]; then rm $result_path; fi
      if [ -f $result_path2 ]; then rm $result_path2; fi
      echo ${alpha[$i]}_${beta[$i]}
      while read line; do
        book_id=$(echo "$line" | cut -f1)
        data_ids=$(echo "$line" | cut -f2-)
        # echo $line
        if [ ! -d rnn_lm/exp_book_$book_id ]; then mkdir -p rnn_lm/exp_book_$book_id; fi
        cp rnn_lm/exp_fed_perbook/epoch-$((epoch-10)).pt rnn_lm/exp_book_$book_id/epoch-$((epoch-10)).pt
        
        if [ $book_id -eq 2002 ]
        then
          batch_size=50
        else
          batch_size=100
        fi
        # echo $batch_size

        ./rnn_lm/train_plm_fed_perbook.py \
          --start-epoch $((epoch-9)) \
          --world-size 3 \
          --num-epochs $((epoch+1)) \
          --use-fp16 0 \
          --embedding-dim 2048 \
          --hidden-dim 2048 \
          --num-layers 3 \
          --batch-size $batch_size \
          --exp-dir rnn_lm/exp_book_$book_id \
          --save-last-epoch true \
          --lm-data-path data/lm_training_bpe_500_userlibri \
          --lm-data-name "$book_id" \
          --alpha ${alpha[$i]} \
          --beta ${beta[$i]}
        
        # if [ "$epoch" != 40 ]; then
        #   if [ -f rnn_lm/exp_$id/epoch-$((epoch-5)).pt ]; then rm rnn_lm/exp_$id/epoch-$((epoch-5)).pt; fi
        # fi

        if [ -f rnn_lm/exp_book_$book_id/epoch-30.pt ]; then rm rnn_lm/exp_book_$book_id/epoch-30.pt; fi
        ./pruned_transducer_stateless5/decode_userlibri.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_book_$book_id \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$data_ids" \
          --book-id "$book_id" \
          --result-path $result_path
          
      done < "$filename"

      lm_list=pruned_transducer_stateless5/results_per_book_fed4_"$epoch"_${alpha[$i]}_${beta[$i]}.txt
      ./utils/average_parameters_fed.py \
        --lm-list ${lm_list} \
        --start-epoch $((epoch+1)) \
        --exp-dir rnn_lm/exp_fed_perbook
      
      while read line; do
        book_id=$(echo "$line" | cut -f1)
        data_ids=$(echo "$line" | cut -f2-)
        ./pruned_transducer_stateless5/decode_userlibri.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_fed_perbook \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$data_ids" \
          --book-id "$book_id" \
          --result-path $result_path2
      done < "$filename"
    done

    echo "Stage 23-1: overfitting stage"

    for epoch in 65 70 75 80; do
      
      result_path=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book_fed4_overfitting_"$epoch".txt

      if [ -f $result_path ]; then rm $result_path; fi
      while read line; do
        book_id=$(echo "$line" | cut -f1)
        data_ids=$(echo "$line" | cut -f2-)
        cp rnn_lm/exp_fed_perbook/epoch-60.pt rnn_lm/exp_book_$book_id/epoch-60.pt
        if [ $book_id -eq 2002 ]
        then
          batch_size=50
        else
          batch_size=100
        fi
        ./rnn_lm/train_plm.py \
          --start-epoch $((epoch-4)) \
          --world-size 3 \
          --num-epochs $((epoch+1)) \
          --use-fp16 0 \
          --embedding-dim 2048 \
          --hidden-dim 2048 \
          --num-layers 3 \
          --batch-size $batch_size \
          --exp-dir rnn_lm/exp_book_$book_id \
          --save-last-epoch true \
          --lm-data-path data/lm_training_bpe_500_userlibri \
          --lm-data-name "$book_id"
        
        ./pruned_transducer_stateless5/decode_userlibri.py \
          --epoch 30 \
          --avg 15 \
          --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
          --max-duration 400 \
          --decoding-method modified_beam_search_lm_shallow_fusion \
          --beam-size 4 \
          --num-encoder-layer 12 \
          --lm-type rnn \
          --lm-scale 0.15 \
          --lm-exp-dir rnn_lm/exp_book_$book_id \
          --lm-epoch $epoch \
          --lm-avg 1 \
          --rnn-lm-num-layers 3 \
          --use-shallow-fusion 1 \
          --rnn-lm-tie-weights 1 \
          --test-id "$data_ids" \
          --book-id "$book_id" \
          --result-path $result_path
      done < "$filename"
    done
  done

  # while read line; do
  #   book_id=$(echo "$line" | cut -f1)
  #   data_ids=$(echo "$line" | cut -f2-)
  #   ./pruned_transducer_stateless5/decode_userlibri.py \
  #     --epoch 30 \
  #     --avg 15 \
  #     --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
  #     --max-duration 400 \
  #     --decoding-method modified_beam_search_lm_shallow_fusion \
  #     --beam-size 4 \
  #     --num-encoder-layer 12 \
  #     --lm-type rnn \
  #     --lm-scale 0.15 \
  #     --lm-exp-dir rnn_lm/exp \
  #     --lm-epoch 30 \
  #     --lm-avg 1 \
  #     --rnn-lm-num-layers 3 \
  #     --use-shallow-fusion 1 \
  #     --rnn-lm-tie-weights 1 \
  #     --test-id "$data_ids" \
  #     --book-id "$book_id" \
  #     --result-path pruned_transducer_stateless5/result_per_book_baseline.txt
  # done < "$filename"
fi

if [ $stage -le 24 ] && [ $stop_stage -ge 24 ]; then
  echo "Stage 24: train and average, finetuning for 10epoch"
  mkdir -p rnn_lm/exp_averaged
  if [ ! -f rnn_lm/exp_averaged/epoch-30.pt ]; then cp rnn_lm/exp/epoch-30.pt rnn_lm/exp_averaged; fi
  filename="data/book_to_data.txt"
  alpha=(1e-4)
  beta=(1e-1)

  for epoch in 40 50 60; do
    result_path=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book2_"$epoch".txt
    result_path2=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book2_"$epoch"_avg.txt

    if [ -f $result_path ]; then rm $result_path; fi
    if [ -f $result_path2 ]; then rm $result_path2; fi

    while read line; do
      book_id=$(echo "$line" | cut -f1)
      data_ids=$(echo "$line" | cut -f2-)
      # echo $line
      if [ ! -d rnn_lm/exp_book_$book_id ]; then mkdir -p rnn_lm/exp_book_$book_id; fi
      cp rnn_lm/exp_averaged/epoch-$((epoch-10)).pt rnn_lm/exp_book_$book_id/epoch-$((epoch-10)).pt
      
      if [ $book_id -eq 2002 ]
      then
        batch_size=50
      else
        batch_size=100
      fi
      # echo $batch_size

      ./rnn_lm/train_plm.py \
        --start-epoch $((epoch-9)) \
        --world-size 3 \
        --num-epochs $((epoch+1)) \
        --use-fp16 0 \
        --embedding-dim 2048 \
        --hidden-dim 2048 \
        --num-layers 3 \
        --batch-size $batch_size \
        --exp-dir rnn_lm/exp_book_$book_id \
        --save-last-epoch true \
        --lm-data-path data/lm_training_bpe_500_userlibri \
        --lm-data-name "$book_id" \
      
      # if [ "$epoch" != 40 ]; then
      #   if [ -f rnn_lm/exp_$id/epoch-$((epoch-5)).pt ]; then rm rnn_lm/exp_$id/epoch-$((epoch-5)).pt; fi
      # fi
      if [ -f rnn_lm/exp_book_$book_id/epoch-30.pt ]; then rm rnn_lm/exp_book_$book_id/epoch-30.pt; fi
      ./pruned_transducer_stateless5/decode_userlibri.py \
        --epoch 30 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
        --max-duration 400 \
        --decoding-method modified_beam_search_lm_shallow_fusion \
        --beam-size 4 \
        --num-encoder-layer 12 \
        --lm-type rnn \
        --lm-scale 0.15 \
        --lm-exp-dir rnn_lm/exp_book_$book_id \
        --lm-epoch $epoch \
        --lm-avg 1 \
        --rnn-lm-num-layers 3 \
        --use-shallow-fusion 1 \
        --rnn-lm-tie-weights 1 \
        --test-id "$data_ids" \
        --book-id "$book_id" \
        --result-path $result_path
        
    done < "$filename"

    lm_list=pruned_transducer_stateless5/results_per_book2_"$epoch".txt
    ./utils/average_parameters_fed.py \
      --lm-list ${lm_list} \
      --start-epoch $((epoch+1)) \
      --exp-dir rnn_lm/exp_averaged
    
    while read line; do
      book_id=$(echo "$line" | cut -f1)
      data_ids=$(echo "$line" | cut -f2-)
      ./pruned_transducer_stateless5/decode_userlibri.py \
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
        --test-id "$data_ids" \
        --book-id "$book_id" \
        --result-path $result_path2
    done < "$filename"
  done
  echo "Stage 20-1: overfitting stage"

  for epoch in 65 70 75 80; do
    
    result_path=/home/lee/Workspace/icefall/egs/librispeech/ASR/pruned_transducer_stateless5/results_per_book2_overfitting_"$epoch".txt

    if [ -f $result_path ]; then rm $result_path; fi
    while read line; do
      book_id=$(echo "$line" | cut -f1)
      data_ids=$(echo "$line" | cut -f2-)
      cp rnn_lm/exp_averaged/epoch-60.pt rnn_lm/exp_book_$book_id/epoch-60.pt
      if [ $book_id -eq 2002 ]
      then
        batch_size=50
      else
        batch_size=100
      fi
      ./rnn_lm/train_plm.py \
        --start-epoch $((epoch-4)) \
        --world-size 3 \
        --num-epochs $((epoch+1)) \
        --use-fp16 0 \
        --embedding-dim 2048 \
        --hidden-dim 2048 \
        --num-layers 3 \
        --batch-size $batch_size \
        --exp-dir rnn_lm/exp_book_$book_id \
        --save-last-epoch true \
        --lm-data-path data/lm_training_bpe_500_userlibri \
        --lm-data-name "$book_id"
      
      ./pruned_transducer_stateless5/decode_userlibri.py \
        --epoch 30 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless5/exp_layer12 \
        --max-duration 400 \
        --decoding-method modified_beam_search_lm_shallow_fusion \
        --beam-size 4 \
        --num-encoder-layer 12 \
        --lm-type rnn \
        --lm-scale 0.15 \
        --lm-exp-dir rnn_lm/exp_book_$book_id \
        --lm-epoch $epoch \
        --lm-avg 1 \
        --rnn-lm-num-layers 3 \
        --use-shallow-fusion 1 \
        --rnn-lm-tie-weights 1 \
        --test-id "$data_ids" \
        --book-id "$book_id" \
        --result-path $result_path
    done < "$filename"
  done
fi