export HYDRA_FULL_ERROR=1
task_names=("square" "threading" "coffee")
seeds=(0 1 2)
for task_name in ${task_names[@]}; do
    for seed in ${seeds[@]}; do
        # sbatch slurm/run_l40s.sbatch \
        echo python train.py \
            --config-name=train_prior.yaml \
            task=mimicgen_rgb_base \
            task.task_name=${task_name} \
            exp_name=mimicgen_corl_final \
            variant_name=3dda_fixed \
            algo=diffuser_actor \
            algo/encoder=hybrid  \
            algo.chunk_size=16 \
            train_dataloader.num_workers=8 \
            algo.policy.use_instruction=false \
            training.n_epochs=251 \
            rollout.interval=1000 \
            seed=${seed}
    done
done


# python train.py \
#     --config-name=train_prior.yaml \
#     task=mimicgen_rgb_base \
#     task.task_name=square \
#     exp_name=debug \
#     variant_name=3dda \
#     algo=diffuser_actor \
#     algo/encoder=hybrid  \
#     algo.chunk_size=16 \
#     train_dataloader.num_workers=8 \
#     algo.policy.use_instruction=false \
#     training.n_epochs=251 \
#     rollout.interval=1000 \
#     logging.mode=disabled \
#     seed=0
