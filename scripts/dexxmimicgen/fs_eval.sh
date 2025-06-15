export HYDRA_FULL_ERROR=1

exp_name=fixed_sweep_dmg_3
eval_exp_name=fixed_sweep_dmg_3
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


for seed in ${seeds[@]}; do
    for variant_name in ${variant_names[@]}; do
        for ego in ${egos[@]}; do
        # original
            echo python evaluate.py \
                exp_name=${eval_exp_name} \
                variant_name=mt_${variant_name}${ego} \
                task=dexmimicge${ego} \
                task.task_name=${task_name} \
                algo=${algo} \
                +overrides.temporal_agg=false +overrides.action_horizon=2 \
                seed=${seed} \
                rollout.rollouts_per_env=100 \
                checkpoint_path=${prefix}/${task_name}/${exp_name}/${variant_name}${ego}/${seed}/
        done
    done
done

