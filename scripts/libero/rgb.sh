export HYDRA_FULL_ERROR=1

uv run train.py \
    --config-name=train_debug.yaml \
    task=libero \
    algo=diffusion_policy \
    algo.chunk_size=8 \
    $@



# # Define arrays for algos and encoders
# algos=("baku" "act" "diffusion_policy")
# encoders=("rgb" "rgbd" "adapt3r" "dp3")

# # Loop through each combination
# for algo in ${algos[@]}; do
#     for encoder in ${encoders[@]}; do
#         python train.py \
#             --config-name=train_debug.yaml \
#             task=libero \
#             algo=$algo \
#             algo/encoder=$encoder \
#             algo.chunk_size=8 \
#             $@
#     done
# done


# python train.py \
#     --config-name=train_debug.yaml \
#     task=libero \
#     algo=act \
#     algo/encoder=rgbd \
#     algo.chunk_size=8 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/vast/quest_v0/experiments/libero/libero_90/libero_sweep_3/act_rgbd/stage_1/multitask_model_epoch_0020.pth \
#     pace_copy=true \
#     rollout.interval=1