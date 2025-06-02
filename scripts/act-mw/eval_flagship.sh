prefix="/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/MT50"
eval_exp_name="act_1"
seeds=(0 1)
algo="act"
exp_names=(
    "act_baseline"
    "act_baseline"
    )
variant_names=(
    "rgb_block_10"
    "rgbd_block_10"
    )
tasks=(
    "metaworld_mt50_rgb"
    "metaworld_mt50_rgbd"
    )


for seed in ${seeds[@]}; do
    for i in {0..1}; do
        exp_name=${exp_names[i]}
        variant_name=${variant_names[i]}
        task=${tasks[i]}
        # echo $exp_name
        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            exp_name=${eval_exp_name} \
            variant_name=${variant_name} \
            task=${task} \
            algo=${algo} \
            rollout.rollouts_per_env=10 \
            rollout.num_parallel_envs=5 \
            checkpoint_path=${prefix}/${algo}/${exp_name}/${variant_name}/${seed}/stage_1 \
            seed=${seed}
    done
done


