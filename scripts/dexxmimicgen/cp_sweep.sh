export HYDRA_FULL_ERROR=1

eval_exp_name=cp_sweep
exp_name=fixed_sweep_dmg_3
algo=diffusion_policy
prefix=/storage/home/hcoda1/1/awilcox31/vast/imitation/experiments/dexmimicgen
seeds=(0 1)
variant_names=(
    adapt3r_bm_concat_ft_abs_eecf
    adapt3r_bm_concat_ft_abs
    adapt3r_bm_separate_ft_abs_eecf
    adapt3r_bm_separate_ft_abs
)
task_name=two_arm_coffee
egos=("n" "n_ego")
epochs=(
    0050
    0100
    0150
    0200
)

for seed in ${seeds[@]}; do
    for variant_name in ${variant_names[@]}; do
        for ego in ${egos[@]}; do
            for epoch in ${epochs[@]}; do
                echo python evaluate.py \
                    exp_name=${eval_exp_name} \
                    variant_name=mt_${variant_name}${ego}_epoch_${epoch} \
                    task=dexmimicge${ego} \
                    task.task_name=${task_name} \
                    algo=${algo} \
                    +overrides.temporal_agg=false +overrides.action_horizon=2 \
                    seed=${seed} \
                    rollout.rollouts_per_env=100 \
                    checkpoint_path=${prefix}/${task_name}/${exp_name}/${variant_name}${ego}/${seed}/multitask_model_epoch_${epoch}.pth
            done
        done
    done
done

