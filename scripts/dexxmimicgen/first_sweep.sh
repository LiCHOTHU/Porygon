export HYDRA_FULL_ERROR=1

# algo_names=(
#     # fm_policy_S
#     # dit_policy_S
#     diffusion_policy
#     fm_policy_M
#     dit_policy_M
#     # fm_policy_L
#     # dit_policy_L 
# )
algo_name=diffusion_policy
encoders=(
    adapt3r
    # rgb
)
# task_names=(
#     square_d1 
#     three_piece_assembly_d1
#     threading_d1
# )
seeds=(0 1)
bimanual_modes=(
    concat
    separate
)
task_name=two_arm_coffee
egos=("n" "n_ego")
# ego="_ego"
# ego=""
# 
for seed in ${seeds[@]}; do
    for ego in ${egos[@]}; do
        for bimanual_mode in ${bimanual_modes[@]}; do
            echo python train.py \
                --config-name=train.yaml \
                exp_name=fixed_sweep_dmg_3 \
                variant_name=adapt3r_bm_${bimanual_mode}_ft_abs${ego} \
                task=dexmimicge${ego} \
                task.task_name=${task_name} \
                algo=${algo_name} \
                algo/encoder=adapt3r_bimanual \
                algo.encoder.bimanual_mode=${bimanual_mode} \
                algo.encoder.do_lang=false \
                algo.chunk_size=16 \
                algo.abs_action=true \
                algo.eecf=false \
                algo.policy.temporal_agg=false \
                algo.encoder.finetune=true \
                rollout.interval=25 \
                training.n_epochs=251 \
                pace_copy=true \
                seed=${seed} \
                $@

            echo python train.py \
                --config-name=train.yaml \
                exp_name=fixed_sweep_dmg_3 \
                variant_name=adapt3r_bm_${bimanual_mode}_ft_abs_eecf${ego} \
                task=dexmimicge${ego} \
                task.task_name=${task_name} \
                algo=${algo_name} \
                algo/encoder=adapt3r_bimanual \
                algo.encoder.bimanual_mode=${bimanual_mode} \
                algo.encoder.do_lang=false \
                algo.chunk_size=16 \
                algo.abs_action=true \
                algo.eecf=true \
                algo.policy.temporal_agg=false \
                algo.encoder.finetune=true \
                rollout.interval=25 \
                training.n_epochs=251 \
                pace_copy=true \
                seed=${seed} \
                $@
        done 

        # NO hand frame

        echo python train.py \
            --config-name=train.yaml \
            exp_name=fixed_sweep_dmg_3 \
            variant_name=adapt3r_ft_no_hf_abs${ego} \
            task=dexmimicge${ego} \
            task.task_name=${task_name} \
            algo=${algo_name} \
            algo/encoder=adapt3r_bimanual \
            algo.encoder.hand_frame=false \
            algo.encoder.bimanual_mode=one \
            algo.chunk_size=16 \
            algo.abs_action=true \
            algo.eecf=false \
            algo.policy.temporal_agg=false \
            algo.encoder.finetune=true \
            rollout.interval=25 \
            training.n_epochs=251 \
            pace_copy=true \
            seed=${seed} \
            $@

        # RGB
        echo python train.py \
            --config-name=train.yaml \
            exp_name=fixed_sweep_dmg_3 \
            variant_name=rgb_abs${ego} \
            task=dexmimicge${ego} \
            task.task_name=${task_name} \
            algo=${algo_name} \
            algo/encoder=rgb \
            algo.chunk_size=16 \
            algo.abs_action=true \
            algo.policy.temporal_agg=false \
            rollout.interval=25 \
            training.n_epochs=251 \
            pace_copy=true \
            seed=${seed} \
            $@
        
        # DP3
        echo python train.py \
            --config-name=train.yaml \
            exp_name=fixed_sweep_dmg_3 \
            variant_name=dp3_abs${ego} \
            task=dexmimicge${ego} \
            task.task_name=${task_name} \
            algo=${algo_name} \
            algo/encoder=dp3 \
            algo.chunk_size=16 \
            algo.abs_action=true \
            algo.policy.temporal_agg=false \
            rollout.interval=25 \
            training.n_epochs=251 \
            pace_copy=true \
            seed=${seed} \
            $@
    done
done

# echo python train.py \
#     --config-name=train_debug.yaml \
#     task=dexmimicgen \
#     task.task_name=square_d1 \
#     algo=fm_policy_M \
#     algo/encoder=rgb \
#     algo.chunk_size=15 \
#     algo.abs_action=false \
#     algo.policy.temporal_agg=false \
#     algo.encoder.finetune=true \
#     rollout.interval=25 \
#     training.n_epochs=501 \
#     pace_copy=true \
#     seed=0
