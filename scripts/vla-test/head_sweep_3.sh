task="libero_90_hybrid"
exp_name="head_sweep_3"

###################### DiT  ##########################

keys=(1 4 16)

for key in ${keys[@]}; do
    sbatch slurm/run_l40s.sbatch python train.py \
        --config-name=train_prior.yaml \
        task=${task} \
        exp_name=${exp_name} \
        variant_name=dit_head_${key}k_no_act_eecf_no_abs \
        pace_copy=true \
        algo=dit_head \
        algo.chunk_size=15 \
        algo.policy.hidden_dim=256 \
        algo.policy.num_layers=4 \
        algo.policy.num_keys=${key} \
        algo.obs_eecf=true \
        algo.act_eecf=false \
        algo.abs_action=false \
        training.save_interval=10 \
        train_dataloader.num_workers=4 \
        rollout.interval=10 \
        task.demos_per_env=50 \
        training.n_epochs=101

    sbatch slurm/run_l40s.sbatch python train.py \
        --config-name=train_prior.yaml \
        task=${task} \
        exp_name=${exp_name} \
        variant_name=dit_head_${key}k \
        pace_copy=true \
        algo=dit_head \
        algo.chunk_size=15 \
        algo.policy.hidden_dim=256 \
        algo.policy.num_layers=4 \
        algo.policy.num_keys=${key} \
        algo.obs_eecf=true \
        algo.act_eecf=true \
        algo.abs_action=true \
        training.save_interval=10 \
        train_dataloader.num_workers=4 \
        rollout.interval=10 \
        task.demos_per_env=50 \
        training.n_epochs=101
done


###################### DiT 2 ########################


sbatch slurm/run_l40s.sbatch python train.py \
    --config-name=train_prior.yaml \
    task=${task} \
    exp_name=${exp_name} \
    variant_name=dit_head_2_no_act_eecf_no_abs \
    pace_copy=true \
    algo=dit_head_2 \
    algo.chunk_size=15 \
    algo.policy.hidden_dim=256 \
    algo.policy.num_layers=4 \
    algo.obs_eecf=true \
    algo.act_eecf=false \
    algo.abs_action=false \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101

sbatch slurm/run_l40s.sbatch python train.py \
    --config-name=train_prior.yaml \
    task=${task} \
    exp_name=${exp_name} \
    variant_name=dit_head_2_no_eecf_no_abs \
    pace_copy=true \
    algo=dit_head_2 \
    algo.chunk_size=15 \
    algo.policy.hidden_dim=256 \
    algo.policy.num_layers=4 \
    algo.obs_eecf=false \
    algo.act_eecf=false \
    algo.abs_action=false \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101

sbatch slurm/run_l40s.sbatch python train.py \
    --config-name=train_prior.yaml \
    task=${task} \
    exp_name=${exp_name} \
    variant_name=dit_head_2_no_act_eecf \
    pace_copy=true \
    algo=dit_head_2 \
    algo.chunk_size=15 \
    algo.policy.hidden_dim=256 \
    algo.policy.num_layers=4 \
    algo.obs_eecf=true \
    algo.act_eecf=false \
    algo.abs_action=false \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101


###################### DA Style #########################


sbatch slurm/run_l40s.sbatch python train.py \
    --config-name=train_prior.yaml \
    task=${task} \
    exp_name=${exp_name} \
    variant_name=da_style_head_no_eecf_fix \
    pace_copy=true \
    algo=da_style_head \
    algo.chunk_size=15 \
    algo.obs_eecf=false \
    algo.act_eecf=false \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101



########################## Baku ################################


sbatch slurm/run_l40s.sbatch python train.py \
    --config-name=train_prior.yaml \
    task=${task} \
    exp_name=${exp_name} \
    variant_name=baku_eecf \
    pace_copy=true \
    algo=baku \
    algo/encoder=hybrid  \
    algo/aug=image \
    algo.chunk_size=10 \
    algo.policy.eecf=true \
    algo.encoder.do_hand_crop=true \
    +algo.encoder.tight_crop=true \
    training.save_interval=20 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101

sbatch slurm/run_l40s.sbatch python train.py \
    --config-name=train_prior.yaml \
    task=${task} \
    exp_name=${exp_name} \
    variant_name=baku_baseline \
    pace_copy=true \
    algo=baku \
    algo/encoder=hybrid  \
    algo/aug=image \
    algo.chunk_size=10 \
    algo.policy.eecf=false \
    algo.abs_action=false \
    algo.encoder.do_hand_crop=true \
    +algo.encoder.tight_crop=true \
    training.save_interval=20 \
    train_dataloader.num_workers=4 \
    rollout.interval=10 \
    task.demos_per_env=50 \
    training.n_epochs=101

