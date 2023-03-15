#!/bin/bash

# Define the path to the directory containing the files
path="data/fbank/"

# Define the output file
output_file="data/userlibri_test_id.txt"

# Iterate through all files in the directory
for file in $path/*.jsonl.gz
do
  # Extract the specific words from the file name
  specific_words=$(echo $file | grep -o -E "(test-clean_[0-9]+|test-other_[0-9]+)" |  tr -d '\n')
  # Append the specific words to the output file
  echo $specific_words >> $output_file
done

sort $output_file | uniq > temp.txt
mv temp.txt $output_file