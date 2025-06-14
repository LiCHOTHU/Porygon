
task_names=("coffee" "square" "threading")
# seeds=(0 1 2)
# task_names=("three_piece_assembly")
export HYDRA_FULL_ERROR=1
seeds=(0 1)
n_epochs=101
for task_name in ${task_names[@]}; do
    for seed in ${seeds[@]}; do
    sbatch slurm/run_l40s.sbatch python train.py \
            --config-name=train_prior.yaml \
            task=mimicgen_hybrid_base \
            exp_name=adapt3r_D1 \
            task.task_name=${task_name} \
            variant_name=adapt3r \
            algo=baku \
            algo/encoder=hybrid  \
            algo.chunk_size=10 \
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
    done
done


python train.py \
    --config-name=train_debug.yaml \
    task=mimicgen_hybrid_base \
    task.task_name=threading \
    algo=baku \
    algo/encoder=hybrid  \
    algo.chunk_size=10 \
    +algo.policy.eecf=true \
    algo.encoder.do_hand_crop=true \
    +algo.encoder.tight_crop=true \
    algo.encoder.do_lang=true \
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=1000 \
    task.demos_per_env=1000\
    training.n_epochs=61
