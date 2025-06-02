prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/libero/libero_90"
eval_exp_name="diffusion_policy_1"
seeds=(0 1)
algo="diffusion_policy"
exp_name="diffusion_policy_no_crop_ablation"
variant_names=(
    "learned_lift_block_16" 
    "no_attn_block_16" 
    "no_eecf_block_16" 
    "no_image_block_16" 
    "no_lang_block_16" 
    "pos_ds_block_16"
    )
task="libero_90_hybrid"


for seed in ${seeds[@]}; do
    for i in {0..5}; do
        variant_name=${variant_names[i]}
        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            exp_name=${eval_exp_name} \
            variant_name=${variant_name} \
            task=${task} \
            algo=${algo} \
            rollout.rollouts_per_env=10 \
            rollout.num_parallel_envs=5 \
            checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1/multitask_model_epoch_0080.pth \
            seed=${seed}
    done
done
