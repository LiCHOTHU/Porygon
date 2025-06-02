
task_names=("coffee" "square" "threading")
# seeds=(0 1 2)
# task_names=("three_piece_assembly")
exp_name=mimicgen_corl
export HYDRA_FULL_ERROR=1
seeds=(0 1)
n_epochs=101
for task_name in ${task_names[@]}; do
    for seed in ${seeds[@]}; do
    sbatch slurm/run_l40s.sbatch python train.py \
            --config-name=train_prior.yaml \
            task=mimicgen_hybrid_base \
            exp_name=${exp_name} \
            task.task_name=${task_name} \
            variant_name=adapt3r \
            algo=baku \
            algo/encoder=hybrid  \
            algo/aug=identity \
            algo.chunk_size=8 \
            +algo.policy.eecf=true \
            algo.encoder.do_hand_crop=true \
            +algo.encoder.tight_crop=true \
            algo.encoder.do_lang=true \
            training.save_interval=10 \
            train_dataloader.num_workers=4 \
            rollout.interval=1000 \
            task.demos_per_env=1000\
            training.n_epochs=${n_epochs} \
            seed=${seed}
            # pace_copy=true  \

         # RGB
        # echo python train.py \
        sbatch slurm/run_l40s.sbatch python train.py \
            --config-name=train_prior.yaml \
            task=mimicgen_rgb_base \
            task.task_name=${task_name} \
            exp_name=${exp_name} \
            variant_name=rgb \
            algo=diffusion_policy \
            algo/aug=identity \
            algo.chunk_size=16 \
            training.save_interval=10 \
            train_dataloader.num_workers=4 \
            rollout.interval=1000 \
            task.demos_per_env=1000 \
            training.n_epochs=${n_epochs} \
            seed=${seed}

        # RGBD
        sbatch slurm/run_l40s.sbatch python train.py \
            --config-name=train_prior.yaml \
            task=mimicgen_rgbd_base \
            task.task_name=${task_name} \
            exp_name=${exp_name} \
            variant_name=rgbd \
            algo=diffusion_policy \
            algo/aug=identity \
            algo.chunk_size=16 \
            training.save_interval=10 \
            train_dataloader.num_workers=4 \
            rollout.interval=1000 \
            task.demos_per_env=1000 \
            training.n_epochs=${n_epochs} \
            seed=${seed}

        # DP3
        # echo python train.py \
        sbatch slurm/run_l40s.sbatch python train.py \
            --config-name=train_prior.yaml \
            task=mimicgen_hybrid_base \
            task.task_name=${task_name} \
            exp_name=${exp_name} \
            variant_name=dp3 \
            algo=diffusion_policy \
            algo/aug=identity \
            algo/encoder=hybrid_dp3 \
            algo.chunk_size=16 \
            training.save_interval=10 \
            train_dataloader.num_workers=4 \
            rollout.interval=1000 \
            task.demos_per_env=1000 \
            training.n_epochs=${n_epochs} \
            seed=${seed}
    done
done