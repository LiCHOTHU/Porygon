#!/bin/bash
export HYDRA_FULL_ERROR=1
exp_name=mimicgen_corl_adapt3r_mix
task_names=("threading")
num_points=(512 1024 2048)
seeds=(0 1)
n_epochs=51
for task_name in ${task_names[@]}; do
    for seed in ${seeds[@]}; do
        for num_point in ${num_points[@]}; do
            echo python train.py \
                --config-name=train_prior.yaml \
                task=mimicgen_hybrid_base \
                exp_name=${exp_name} \
                task.task_name=${task_name} \
                variant_name=adapt3r_${num_point}_finetune \
                algo=diffusion_policy \
                algo/encoder=hybrid  \
                algo.encoder.num_points=${num_point} \
                algo.chunk_size=8 \
                +algo.policy.eecf=true \
                algo.encoder.do_hand_crop=true \
                +algo.encoder.tight_crop=true \
                algo.encoder.do_lang=true \
                training.save_interval=10 \
                train_dataloader.num_workers=4 \
                rollout.interval=1000 \
                seed=${seed}

            echo python train.py \
                --config-name=train_prior.yaml \
                task=mimicgen_hybrid_base \
                exp_name=${exp_name} \
                task.task_name=${task_name} \
                variant_name=adapt3r_${num_point}_no-finetune \
                algo=diffusion_policy \
                algo/encoder=hybrid  \
                algo.chunk_size=8 \
                +algo.policy.eecf=true \
                algo.encoder.do_hand_crop=true \
                algo.encoder.num_points=${num_point} \
                algo.encoder.finetune=false \
                +algo.encoder.tight_crop=true \
                algo.encoder.do_lang=true \
                training.save_interval=10 \
                train_dataloader.num_workers=4 \
                rollout.interval=1000 \
                seed=${seed}

            echo python train.py \
                --config-name=train_prior.yaml \
                task=mimicgen_hybrid_base \
                exp_name=${exp_name} \
                task.task_name=${task_name} \
                variant_name=adapt3r_${num_point}_finetune_no-hand-crop \
                algo=diffusion_policy \
                algo/encoder=hybrid  \
                algo.encoder.num_points=${num_point} \
                algo.chunk_size=8 \
                +algo.policy.eecf=true \
                algo.encoder.do_hand_crop=false \
                +algo.encoder.tight_crop=true \
                algo.encoder.do_lang=true \
                training.save_interval=10 \
                train_dataloader.num_workers=4 \
                rollout.interval=1000 \
                seed=${seed}

            echo python train.py \
                --config-name=train_prior.yaml \
                task=mimicgen_hybrid_base \
                exp_name=${exp_name} \
                task.task_name=${task_name} \
                variant_name=adapt3r_${num_point}_no-finetune_no-hand-crop \
                algo=diffusion_policy \
                algo/encoder=hybrid  \
                algo.chunk_size=8 \
                +algo.policy.eecf=true \
                algo.encoder.do_hand_crop=false \
                algo.encoder.num_points=${num_point} \
                algo.encoder.finetune=false \
                +algo.encoder.tight_crop=true \
                algo.encoder.do_lang=true \
                training.save_interval=10 \
                train_dataloader.num_workers=4 \
                rollout.interval=1000 \
                seed=${seed}
        done
    done 
done 