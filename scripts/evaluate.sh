#!/bin/bash
# Example usage of the Evaluate.py script with quantization enabled
# Make sure to adjust the parameters as needed

python ../Evaluate.py \
  --token "YOUR_HF_TOKEN" \
  --outDir "../results" \
  --indicesMod 1 \
  --indicesRemainder 0 \
  --model_name "QCRI/Fanar-1-9B" \
  --tokenizer_name "QCRI/Fanar-1-9B" \
  --device "cuda" \
  --batch_size 1 \
  --max_length 512 \
  --enable_quantization true\
  --data_directory "Abderraouf000/IslamTrust-benchmark" \
  --bits 4\
  --language "Arabic"



# Replace YOUR_HF_TOKEN with your actual Hugging Face token