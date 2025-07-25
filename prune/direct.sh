#!/bin/bash

python direct_cot.py \
     --model_name "model_name" \
     --result_file "result_file" \
     --data_path "data_path" \
     --output_dir "output_dir" \
     --temperature 0.0 \
     --max_tokens 16384 \
     --top_p 1.0 \
     --key "key" \
     --url "url" \
     --min_index 0 \
     --max_index 10000 \
     --save > generate-direct-cot.out 2>&1 &


