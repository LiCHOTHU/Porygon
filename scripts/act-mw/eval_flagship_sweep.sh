prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/MT50"
eval_exp_name="act_sweep"
seeds=(0)
algo="act"
epochs=(100 200 300 400 500)
exp_name="act_no_crop_3"
variant_names=(
    "flagship_eecf_crop_fixed_block_10"
    "flagship_eecf_crop_fixed_block_15"
    )
task="metaworld_mt50_hybrid"


for seed in ${seeds[@]}; do
    for variant_name in ${variant_names[@]}; do
        for epoch in ${epochs[@]}; do
            sbatch slurm/run_v100.sbatch python evaluate.py \
                exp_name=${eval_exp_name} \
                variant_name=${variant_name}_epoch_${epoch} \
                task=${task} \
                algo=${algo} \
                rollout.rollouts_per_env=10 \
                rollout.num_parallel_envs=5 \
                checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0${epoch}.pth \
                seed=${seed}
        done
    done
done


