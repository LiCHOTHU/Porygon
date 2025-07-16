
exp_name=baselines_D1_no_aug
export HYDRA_FULL_ERROR=1
seeds=(0 1)
task_names=("coffee" "square" "threading")
n_epochs=101
for seed in ${seeds[@]}; do
    for task_name in ${task_names[@]}; do
        # RGB
        # echo python train.py \
        sbatch slurm/run_l40s.sbatch python train.py \
            --config-name=train_prior.yaml \
            task=mimicgen_rgb_base \
            task.task_name=${task_name} \
            exp_name=${exp_name} \
            variant_name=rgb \
            algo=baku \
            algo/aug=identity \
            algo.chunk_size=15 \
            training.save_interval=10 \
            train_dataloader.num_workers=4 \
            rollout.interval=1000 \
            task.demos_per_env=1000 \
            training.n_epochs=${n_epochs} \
            seed=${seed}

        # RGBD
        # echo python train.py \
        sbatch slurm/run_l40s.sbatch python train.py \
            --config-name=train_prior.yaml \
            task=mimicgen_rgbd_base \
            task.task_name=${task_name} \
            exp_name=${exp_name} \
            variant_name=rgbd \
            algo=baku \
            algo/aug=identity \
            algo.chunk_size=15 \
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
            algo=baku \
            algo/aug=identity \
            algo/encoder=hybrid_dp3 \
            algo.chunk_size=15 \
            training.save_interval=10 \
            train_dataloader.num_workers=4 \
            rollout.interval=1000 \
            task.demos_per_env=1000 \
            training.n_epochs=${n_epochs} \
            seed=${seed}
    done
done