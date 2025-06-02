export HYDRA_FULL_ERROR=1

# Define arrays for algos and encoders
algos=("baku" "act" "diffusion_policy")
encoders=("rgb" "rgbd" "adapt3r" "dp3")

# Loop through each combination
# for algo in ${algos[@]}; do
#     for encoder in ${encoders[@]}; do
#         sbatch slurm/run_l40s.sbatch python train.py \
#             --config-name=train.yaml \
#             exp_name=libero_sweep_3 \
#             variant_name=${algo}_${encoder} \
#             task=libero \
#             algo=$algo \
#             algo/encoder=$encoder \
#             algo.chunk_size=8 \
#             pace_copy=true
#     done
# done

sbatch slurm/run_l40s.sbatch python train.py \
    --config-name=train.yaml \
    exp_name=libero_sweep_3 \
    variant_name=diffuser_actor \
    task=libero \
    algo=diffuser_actor \
    pace_copy=true