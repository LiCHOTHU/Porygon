for ((i=0; i<90; i+=2)); do
    sbatch slurm/run_rtx6000.sbatch python scripts/fix_libero_data.py task=libero_90_data +task_nums=[$i,$((i + 1))]
done


