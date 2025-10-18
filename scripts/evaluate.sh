#!/bin/bash
# Example usage of the Evaluate.py script with quantization enabled
# Make sure to adjust the parameters as needed

python ../Evaluate.py \
  --token "YOUR_HF_TOKEN" \
  --outDir "../results" \
  --indicesMod 1 \
  --indicesRemainder 0 \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --tokenizer_name "meta-llama/Llama-3.1-8B-Instruct" \
  --device "cuda" \
  --batch_size 1 \
  --max_length 512 \
  --enable_quantization \
  --data_directory "IslamTrust-benchmark/benchmark-datasets/IslamBench-Arabic.csv" \
  --bits 4

# Replace YOUR_HF_TOKEN with your actual Hugging Face token