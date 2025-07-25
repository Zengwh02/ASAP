CUDA_DEVICE=0
MODEL_NAME_OR_PATH="model_name"
DATA="data_path"
BATCH_SIZE=2000
LOG_DIR="./logs"
TEMPLATE="ds"
RESULT_DIR="./results"
BUDGET=4096
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python instruct.py \
  --model ${MODEL_NAME_OR_PATH} \
  --data ${DATA} \
  --batch_size ${BATCH_SIZE} \
  --output_dir ${RESULT_DIR} \
  --save_dir ${RESULT_DIR}.json \
  --template ${TEMPLATE} \
  --budget ${BUDGET} > ${LOG_DIR}/${BUDGET}.out 2>&1
