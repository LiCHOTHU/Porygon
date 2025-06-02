exp_name=baselines_D1
export HYDRA_FULL_ERROR=1
seeds=(0 1)
prefix=/storage/home/hcoda1/7/mghanem8/p-agarg35-0/codes/quest_v0/experiments
task_names=("coffee" "square" "threading")
n_epochs=0050

for seed in ${seeds[@]}; do
    for task_name in ${task_names[@]}; do
        # RGB
        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            exp_name=${exp_name} \
            variant_name=rgb_${n_epochs} \
            task=mimicgen_rgb_base \
            task.task_name=${task_name} \
            algo=baku \
            seed=${seed} \
            checkpoint_path=${prefix}/mimicgen/${task_name}/baku/${exp_name}/rgb/${seed}/stage_1/multitask_model_epoch_${n_epochs}.pth

        # RGBD
        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            exp_name=${exp_name} \
            variant_name=rgbd_${n_epochs} \
            task=mimicgen_rgbd_base \
            task.task_name=${task_name} \
            algo=baku \
            seed=${seed} \
            checkpoint_path=${prefix}/mimicgen/${task_name}/baku/${exp_name}/rgbd/${seed}/stage_1/multitask_model_epoch_${n_epochs}.pth

        # DP3
        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            exp_name=${exp_name} \
            variant_name=dp3_${n_epochs} \
            task=mimicgen_hybrid_base \
            algo/encoder=hybrid_dp3 \
            task.task_name=${task_name} \
            algo=baku \
            seed=${seed} \
            checkpoint_path=${prefix}/mimicgen/${task_name}/baku/${exp_name}/dp3/${seed}/stage_1/multitask_model_epoch_${n_epochs}.pth
    done
done