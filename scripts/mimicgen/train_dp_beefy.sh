export HYDRA_FULL_ERROR=1
task_names=("threading")
exp_name=mimicgen_corl
seeds=(0 1)
n_epochs=51
for task_name in ${task_names[@]}; do
    for seed in ${seeds[@]}; do
        # sbatch slurm/run_l40s.sbatch python train.py \
        #     --config-name=train_prior.yaml \
        #     task=mimicgen_hybrid_base \
        #     exp_name=${exp_name} \
        #     task.task_name=${task_name} \
        #     variant_name=adapt3r_beefy_cs_8 \
        #     algo=diffusion_policy \
        #     algo/encoder=beefy  \
        #     algo.chunk_size=8 \
        #     +algo.policy.eecf=true \
        #     algo.encoder.do_hand_crop=true \
        #     +algo.encoder.tight_crop=true \
        #     algo.encoder.do_lang=false \
        #     training.save_interval=10 \
        #     train_dataloader.num_workers=4 \
        #     rollout.interval=1000 \
        #     task.demos_per_env=1000\
        #     training.n_epochs=${n_epochs} \
        #     seed=${seed}

        # sbatch slurm/run_l40s.sbatch python train.py \
        #     --config-name=train_prior.yaml \
        #     task=mimicgen_hybrid_base \
        #     exp_name=${exp_name} \
        #     task.task_name=${task_name} \
        #     variant_name=adapt3r_beefy_cs_16 \
        #     algo=diffusion_policy \
        #     algo/encoder=beefy  \
        #     algo.chunk_size=16 \
        #     +algo.policy.eecf=true \
        #     algo.encoder.do_hand_crop=true \
        #     +algo.encoder.tight_crop=true \
        #     algo.encoder.do_lang=false \
        #     training.save_interval=10 \
        #     train_dataloader.num_workers=4 \
        #     rollout.interval=1000 \
        #     task.demos_per_env=1000\
        #     training.n_epochs=${n_epochs} \
        #     seed=${seed}

        # sbatch slurm/run_l40s.sbatch python train.py \
        #     --config-name=train_prior.yaml \
        #     task=mimicgen_hybrid_base \
        #     exp_name=${exp_name} \
        #     task.task_name=${task_name} \
        #     variant_name=adapt3r_beefy_cs_8_ft \
        #     algo=diffusion_policy \
        #     algo/encoder=beefy  \
        #     algo.encoder.finetune=true \
        #     algo.chunk_size=8 \
        #     +algo.policy.eecf=true \
        #     algo.encoder.do_hand_crop=true \
        #     +algo.encoder.tight_crop=true \
        #     algo.encoder.do_lang=false \
        #     training.save_interval=10 \
        #     train_dataloader.num_workers=4 \
        #     rollout.interval=1000 \
        #     task.demos_per_env=1000\
        #     training.n_epochs=${n_epochs} \
        #     seed=${seed}
    done
done

# python train.py \
#     --config-name=train_debug.yaml \
#     task=mimicgen_hybrid_base \
#     task.task_name=threading \
#     algo=diffusion_policy \
#     algo/encoder=beefy  \
#     algo.chunk_size=8 \
#     training.save_interval=10 \
#     train_dataloader.num_workers=4 \
#     rollout.interval=1000 \
#     task.demos_per_env=1000\
#     training.n_epochs=50


# python train.py \
#     --config-name=train_debug.yaml \
#     task=mimicgen_hybrid_base \
#     task.task_name=threading \
#     algo=diffusion_policy \
#     algo/encoder=beefy_2  \
#     algo.chunk_size=8 \
#     training.save_interval=10 \
#     train_dataloader.num_workers=4 \
#     rollout.interval=1000 \
#     task.demos_per_env=1000\
#     training.n_epochs=50
