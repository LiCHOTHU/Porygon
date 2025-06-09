export HYDRA_FULL_ERROR=1

python train.py \
    --config-name=train_debug.yaml \
    task=dexmimicgen \
    algo=diffusion_policy \
    algo/encoder=rgb  \
    algo.chunk_size=8 \
    algo.abs_action=true \
    algo.temporal_agg=false \
    $@

