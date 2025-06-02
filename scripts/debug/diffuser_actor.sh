export HYDRA_FULL_ERROR=1

python train.py \
    --config-name=train_debug.yaml \
    task=libero \
    algo=diffuser_actor \
    algo.chunk_size=8 \
    $@


