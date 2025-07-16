for ((i=0; i<90; i+=2)); do
    sbatch slurm/run_rtx6000.sbatch uv run scripts/process_libero_data.py task=libero +task_nums=[$i,$((i + 1))]
done


# uv run scripts/process_libero_data.py task=libero