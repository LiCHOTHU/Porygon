export HYDRA_FULL_ERROR=1
# export CUDA_LAUNCH_BLOCKING=1

python train.py \
    --config-name=train_debug.yaml \
    task=dexmimicgen_ego \
    algo=diffusion_policy \
    algo/encoder=adapt3r_bimanual  \
    algo.encoder.do_lang=false \
    algo.chunk_size=8 \
    algo.abs_action=true \
    algo.temporal_agg=false \
    algo.encoder.finetune=true \
    $@

