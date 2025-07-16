export HYDRA_FULL_ERROR=1
task_names=("square" "threading")
exp_name=mimicgen_corl_new_hand_crop
seeds=(0 1)
n_epochs=501
for task_name in ${task_names[@]}; do
    for seed in ${seeds[@]}; do
        # sbatch slurm/run_l40s.sbatch 
        # python train.py \
        #     --config-name=train_prior.yaml \
        #     task=mimicgen_hybrid_base \
        #     exp_name=${exp_name} \
        #     task.task_name=${task_name} \
        #     variant_name=adapt3r \
        #     algo=diffusion_policy \
        #     algo/encoder=hybrid  \
        #     algo.chunk_size=8 \
        #     +algo.policy.eecf=true \
        #     algo.encoder.do_hand_crop=true \
        #     +algo.encoder.tight_crop=true \
        #     algo.encoder.do_lang=true \
        #     training.save_interval=10 \
        #     train_dataloader.num_workers=4 \
        #     rollout.interval=1000 \
        #     task.demos_per_env=1000\
        #     training.n_epochs=${n_epochs} \
        #     seed=${seed}

        # RGB
        # sbatch slurm/run_l40s.sbatch python train.py \
        echo python train.py \
            --config-name=train_prior.yaml \
            task=mimicgen_rgb_base \
            task.task_name=${task_name} \
            exp_name=${exp_name} \
            variant_name=rgb_500_epochs \
            algo=diffusion_policy \
            algo.chunk_size=16 \
            training.save_interval=10 \
            train_dataloader.num_workers=4 \
            rollout.interval=1000 \
            task.demos_per_env=1000 \
            training.n_epochs=${n_epochs} \
            logging.mode=disabled \
            seed=${seed}

        # # RGBD
        # sbatch slurm/run_l40s.sbatch python train.py \
        #     --config-name=train_prior.yaml \
        #     task=mimicgen_rgbd_base \
        #     task.task_name=${task_name} \
        #     exp_name=${exp_name} \
        #     variant_name=rgbd \
        #     algo=diffusion_policy \
        #     algo.chunk_size=16 \
        #     training.save_interval=10 \
        #     train_dataloader.num_workers=4 \
        #     rollout.interval=1000 \
        #     task.demos_per_env=1000 \
        #     training.n_epochs=${n_epochs} \
        #     seed=${seed}

        # # DP3
        # # echo python train.py \
        # sbatch slurm/run_l40s.sbatch python train.py \
        #     --config-name=train_prior.yaml \
        #     task=mimicgen_hybrid_base \
        #     task.task_name=${task_name} \
        #     exp_name=${exp_name} \
        #     variant_name=dp3 \
        #     algo=diffusion_policy \
        #     algo/encoder=hybrid_dp3 \
        #     algo.chunk_size=16 \
        #     training.save_interval=10 \
        #     train_dataloader.num_workers=4 \
        #     rollout.interval=1000 \
        #     task.demos_per_env=1000 \
        #     training.n_epochs=${n_epochs} \
        #     seed=${seed}
    done
done


python train.py \
    --config-name=train_debug.yaml \
    task=mimicgen_rgb_base \
    task.task_name=square \
    algo=diffusion_policy \
    algo.chunk_size=16 \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=1000 \
    task.demos_per_env=1000 \
    training.n_epochs=1000