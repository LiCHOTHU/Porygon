export HYDRA_FULL_ERROR=1
task_names=("square")
exp_name=mimicgen_corl
seeds=(0 1)
n_epochs=51
for task_name in ${task_names[@]}; do
    for seed in ${seeds[@]}; do
        echo python train.py \
            --config-name=train_prior.yaml \
            task=mimicgen_hybrid_base \
            exp_name=${exp_name} \
            task.task_name=${task_name} \
            variant_name=adapt3r_no-hand-crop \
            algo=diffusion_policy \
            algo/encoder=hybrid \
            algo.chunk_size=8 \
            +algo.policy.eecf=true \
            algo.encoder.do_hand_crop=false \
            +algo.encoder.tight_crop=true \
            algo.encoder.do_lang=true \
            training.save_interval=10 \
            train_dataloader.num_workers=4 \
            rollout.interval=1000 \
            training.n_epochs=${n_epochs} \
            seed=${seed}

        echo python train.py \
            --config-name=train_prior.yaml \
            task=mimicgen_hybrid_base \
            exp_name=${exp_name} \
            task.task_name=${task_name} \
            variant_name=adapt3r_hand-crop-new \
            algo=diffusion_policy \
            algo/encoder=hybrid \
            algo.chunk_size=8 \
            +algo.policy.eecf=true \
            algo.encoder.do_hand_crop=true \
            +algo.encoder.tight_crop=true \
            algo.encoder.do_lang=true \
            training.save_interval=10 \
            train_dataloader.num_workers=4 \
            rollout.interval=1000 \
            training.n_epochs=${n_epochs} \
            seed=${seed}

        echo python train.py \
            --config-name=train_prior.yaml \
            task=mimicgen_hybrid_base \
            exp_name=${exp_name} \
            task.task_name=${task_name} \
            variant_name=adapt3r_hand-crop-old \
            algo=diffusion_policy \
            algo/encoder=hybrid \
            algo.chunk_size=8 \
            +algo.policy.eecf=true \
            algo.encoder.do_hand_crop=true \
            +algo.encoder.tight_crop=true \
            algo.encoder.do_lang=true \
            algo.encoder.use_old_hand_frame_crop=true \
            training.save_interval=10 \
            train_dataloader.num_workers=4 \
            rollout.interval=1000 \
            training.n_epochs=${n_epochs} \
            seed=${seed}
    done
done