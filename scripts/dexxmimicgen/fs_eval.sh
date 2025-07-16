export HYDRA_FULL_ERROR=1

exp_name=fixed_sweep_dmg_4
eval_exp_name=fixed_sweep_dmg_4
algo=diffusion_policy
prefix=/storage/home/hcoda1/1/awilcox31/vast/imitation/experiments/dexmimicgen
seeds=(0 1)
variant_names=(
    adapt3r_bm_concat_ft_abs_eecf
    adapt3r_bm_concat_ft_abs
    adapt3r_bm_separate_ft_abs_eecf
    adapt3r_bm_separate_ft_abs
    adapt3r_ft_no_hf_abs
    dp3_abs
    rgb_abs
)
task_name=two_arm_coffee
egos=("n" "n_ego")
checkpoints=(
    0020
    0040
    0060
    0080
)


for seed in ${seeds[@]}; do
    for variant_name in ${variant_names[@]}; do
        for ego in ${egos[@]}; do
            for checkpoint in ${checkpoints[@]}; do
            echo python evaluate.py \
                exp_name=${eval_exp_name} \
                variant_name=mt_${variant_name}${ego}_${checkpoint} \
                task=dexmimicge${ego} \
                task.task_name=${task_name} \
                algo=${algo} \
                +overrides.temporal_agg=false +overrides.action_horizon=2 \
                seed=${seed} \
                rollout.rollouts_per_env=100 \
                checkpoint_path=${prefix}/${task_name}/${exp_name}/${variant_name}${ego}/${seed}/multitask_model_epoch_${checkpoint}.pth
            done
        done
    done
done

