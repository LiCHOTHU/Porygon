export HYDRA_FULL_ERROR=1

task_names=(square_d1 coffee_d1 threading_d1)
abs_actions=(true false)

for task_name in ${task_names[@]}; do
    for abs_action in ${abs_actions[@]}; do
        sbatch slurm/run_l40s.sbatch uv run train.py \
            --config-name=train.yaml \
            exp_name=mimicgen_sweep \
            variant_name=${task_name}_aa_${abs_action} \
            task=mimicgen \
            task.task_name=${task_name} \
            algo=diffusion_policy \
            algo/encoder=rgb  \
            algo.chunk_size=8 \
            algo.abs_action=${abs_action} \
            algo.policy.temporal_agg=false \
            rollout.interval=25 \
            training.n_epochs=301 \
            pace_copy=true \
            $@
    done
done

scancel 4638874  
scancel 4638875  
scancel 4638876  
scancel 4638877  
scancel 4638878  
scancel 4638879  