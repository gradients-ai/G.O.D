FROM diagonalge/kohya_latest:latest

RUN mkdir -p /dataset/configs \
    /dataset/outputs \
    /dataset/images && \
    chmod -R 777 /dataset

ENV CONFIG_DIR="/dataset/configs"
ENV OUTPUT_DIR="/dataset/outputs"
ENV DATASET_DIR="/dataset/images"

CMD accelerate launch --dynamo_backend no --dynamo_mode default --mixed_precision bf16 --num_processes 1 --num_machines 1 --num_cpu_threads_per_process 2 /app/sd-scripts/${BASE_MODEL}_train_network.py --config_file ${CONFIG_DIR}/${JOB_ID}.toml