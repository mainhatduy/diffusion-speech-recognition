uv run python scripts/data-preprocess/precompute_embeddings.py \
    --output_dir precomputed_data \
    --audio_encoder_name UsefulSensors/moonshine-streaming-medium \
    --pretrained FacebookAI/xlm-roberta-base \
    --batch_size 32 \
    --max_length 128 \
    --num_workers 8 \
    --dtype float16
