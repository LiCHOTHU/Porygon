export HYDRA_FULL_ERROR=1
task_names=(
    "square_d1" 
    "threading_d1" 
    "coffee_d1"
    )
exp_name=mimicgen_oops
# seeds=(0 1 2 3 4)
seeds=(0 1)
n_epochs=251


        
for task_name in ${task_names[@]}; do
    for seed in ${seeds[@]}; do
        # RGB
        echo uv run train.py \
            --config-name=train.yaml \
            task=mimicgen \
            task.task_name=${task_name} \
            exp_name=${exp_name} \
            variant_name=rgb \
            algo=diffusion_policy \
            algo/encoder=rgb \
            algo.chunk_size=16 \
            training.save_interval=10 \
            train_dataloader.num_workers=4 \
            rollout.interval=25 \
            training.n_epochs=${n_epochs} \
            pace_copy=true \
            seed=${seed}

        # RGBD
        echo uv run train.py \
            --config-name=train.yaml \
            task=mimicgen \
            task.task_name=${task_name} \
            exp_name=${exp_name} \
            variant_name=rgbd \
            algo=diffusion_policy \
            algo/encoder=rgbd \
            algo.chunk_size=16 \
            training.save_interval=10 \
            train_dataloader.num_workers=4 \
            rollout.interval=25 \
            training.n_epochs=${n_epochs} \
            pace_copy=true \
            seed=${seed}

        # DP3
        echo uv run train.py \
            --config-name=train.yaml \
            task=mimicgen \
            task.task_name=${task_name} \
            exp_name=${exp_name} \
            variant_name=dp3 \
            algo=diffusion_policy \
            algo/encoder=dp3 \
            algo.chunk_size=16 \
            training.save_interval=10 \
            train_dataloader.num_workers=4 \
            rollout.interval=25 \
            training.n_epochs=${n_epochs} \
            pace_copy=true \
            seed=${seed}

        # Adapt3r
        echo uv run train.py \
            --config-name=train.yaml \
            task=mimicgen \
            exp_name=${exp_name} \
            task.task_name=${task_name} \
            variant_name=adapt3r_ft \
            algo=diffusion_policy \
            algo/encoder=adapt3r  \
            algo.chunk_size=16 \
            algo.encoder.finetune=true \
            training.save_interval=10 \
            train_dataloader.num_workers=4 \
            rollout.interval=25 \
            training.n_epochs=${n_epochs} \
            pace_copy=true \
            seed=${seed}
    done
done

