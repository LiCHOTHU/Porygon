export HYDRA_FULL_ERROR=1
task_names=(
    # "square" 
    # "threading" 
    "coffee"
    )
exp_name=mimicgen_corl_final
seeds=(0 1 2 3 4)
n_epochs=251
for seed in ${seeds[@]}; do 
    
    # # sbatch slurm/run_l40s.sbatch 
    # echo python train.py \
    #     --config-name=train_prior.yaml \
    #     task=mimicgen_hybrid_base \
    #     exp_name=${exp_name} \
    #     task.task_name=square \
    #     variant_name=adapt3r \
    #     algo=diffusion_policy \
    #     algo/encoder=hybrid  \
    #     algo.chunk_size=8 \
    #     +algo.policy.eecf=true \
    #     algo.encoder.do_hand_crop=true \
    #     +algo.encoder.tight_crop=true \
    #     algo.encoder.do_lang=true \
    #     algo.encoder.use_old_hand_frame_crop=true \
    #     algo.encoder.finetune=true \
    #     training.save_interval=10 \
    #     train_dataloader.num_workers=4 \
    #     rollout.interval=1000 \
    #     task.demos_per_env=1000\
    #     training.n_epochs=${n_epochs} \
    #     seed=${seed}

    # # sbatch slurm/run_l40s.sbatch
    # echo python train.py \
    #     --config-name=train_prior.yaml \
    #     task=mimicgen_hybrid_base \
    #     exp_name=${exp_name} \
    #     task.task_name=threading \
    #     variant_name=adapt3r \
    #     algo=diffusion_policy \
    #     algo/encoder=hybrid  \
    #     algo.chunk_size=8 \
    #     +algo.policy.eecf=true \
    #     algo.encoder.do_hand_crop=true \
    #     +algo.encoder.tight_crop=true \
    #     algo.encoder.do_lang=true \
    #     algo.encoder.use_old_hand_frame_crop=false \
    #     algo.encoder.finetune=true \
    #     training.save_interval=10 \
    #     train_dataloader.num_workers=4 \
    #     rollout.interval=1000 \
    #     task.demos_per_env=1000\
    #     training.n_epochs=${n_epochs} \
    #     seed=${seed}

    # sbatch slurm/run_l40s.sbatch 
    echo python train.py \
        --config-name=train_prior.yaml \
        task=mimicgen_hybrid_base \
        exp_name=${exp_name} \
        task.task_name=coffee \
        variant_name=adapt3r_ft \
        algo=diffusion_policy \
        algo/encoder=hybrid  \
        algo.chunk_size=8 \
        +algo.policy.eecf=true \
        algo.encoder.do_hand_crop=true \
        +algo.encoder.tight_crop=true \
        algo.encoder.do_lang=true \
        algo.encoder.use_old_hand_frame_crop=false \
        algo.encoder.finetune=false \
        training.save_interval=10 \
        train_dataloader.num_workers=4 \
        rollout.interval=1000 \
        task.demos_per_env=1000\
        training.n_epochs=${n_epochs} \
        seed=${seed}

done


        
# for task_name in ${task_names[@]}; do
#     for seed in ${seeds[@]}; do
#         # RGB
#         # sbatch slurm/run_l40s.sbatch python train.py \
#         echo python train.py \
#             --config-name=train_prior.yaml \
#             task=mimicgen_rgb_base \
#             task.task_name=${task_name} \
#             exp_name=${exp_name} \
#             variant_name=rgb \
#             algo=diffusion_policy \
#             algo.chunk_size=16 \
#             training.save_interval=10 \
#             train_dataloader.num_workers=4 \
#             rollout.interval=1000 \
#             task.demos_per_env=1000 \
#             training.n_epochs=${n_epochs} \
#             logging.mode=disabled \
#             seed=${seed}

#         # RGBD
#         # sbatch slurm/run_l40s.sbatch
#         echo python train.py \
#             --config-name=train_prior.yaml \
#             task=mimicgen_rgbd_base \
#             task.task_name=${task_name} \
#             exp_name=${exp_name} \
#             variant_name=rgbd \
#             algo=diffusion_policy \
#             algo.chunk_size=16 \
#             training.save_interval=10 \
#             train_dataloader.num_workers=4 \
#             rollout.interval=1000 \
#             task.demos_per_env=1000 \
#             training.n_epochs=${n_epochs} \
#             seed=${seed}

#         # DP3
#         # sbatch slurm/run_l40s.sbatch python train.py \
#         echo python train.py \
#             --config-name=train_prior.yaml \
#             task=mimicgen_hybrid_base \
#             task.task_name=${task_name} \
#             exp_name=${exp_name} \
#             variant_name=dp3 \
#             algo=diffusion_policy \
#             algo/encoder=hybrid_dp3 \
#             algo.chunk_size=16 \
#             training.save_interval=10 \
#             train_dataloader.num_workers=4 \
#             rollout.interval=1000 \
#             task.demos_per_env=1000 \
#             training.n_epochs=${n_epochs} \
#             seed=${seed}
#     done
# done