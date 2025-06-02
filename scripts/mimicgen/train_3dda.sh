
exp_name=mimicgen_corl_final
export HYDRA_FULL_ERROR=1
seeds=(0 1 2 3 4)
task_names=(
    "coffee"
    "square"
    "threading"
)
n_epochs=251
for seed in ${seeds[@]}; do
    for task_name in ${task_names[@]}; do
        # RGB
        # echo python train.py \
        echo python train.py \
            --config-name=train_prior.yaml \
            task=mimicgen_hybrid_base \
            task.task_name=${task_name} \
            exp_name=${exp_name} \
            variant_name=3dda \
            algo=diffuser_actor \
            algo.chunk_size=16 \
            algo.policy.use_instruction=false \
            training.save_interval=10 \
            train_dataloader.num_workers=4 \
            rollout.interval=1000 \
            task.demos_per_env=1000 \
            training.n_epochs=${n_epochs} \
            seed=${seed}

    done
done


# python train.py \
#     --config-name=train_debug.yaml \
#     task=mimicgen_hybrid_base \
#     task.task_name=coffee \
#     algo=diffuser_actor \
#     algo.chunk_size=16 \
#     algo.policy.use_instruction=false \
#     training.save_interval=10 \
#     train_dataloader.num_workers=4 \
#     rollout.interval=1000 \
#     task.demos_per_env=1000 \
#     training.n_epochs=251 
# python train.py \
#     --config-name=train_debug.yaml \
#     task=mimicgen_hybrid_base \
#     task.task_name=threading \
#     algo=diffuser_actor \
#     algo.chunk_size=16 \
#     algo.policy.use_instruction=false \
#     training.save_interval=10 \
#     train_dataloader.num_workers=4 \
#     rollout.interval=1000 \
#     task.demos_per_env=1000 \
#     training.n_epochs=100 

scancel 4209556  
scancel 4209553  
scancel 4209554  
scancel 4209555  
scancel 4209552  