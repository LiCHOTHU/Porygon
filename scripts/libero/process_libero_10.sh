# sbatch slurm/run_l40s.sbatch python scripts/fix_libero_data.py  task=libero_10_data +setting_filter=KITCHEN
# sbatch slurm/run_l40s.sbatch python scripts/fix_libero_data.py  task=libero_10_data +setting_filter=LIVING_ROOM
# sbatch slurm/run_l40s.sbatch python scripts/fix_libero_data.py  task=libero_10_data +setting_filter=STUDY

# sbatch slurm/run_l40s.sbatch python scripts/fix_libero_data.py  task=libero_90_data +setting_filter=KITCHEN
# sbatch slurm/run_l40s.sbatch python scripts/fix_libero_data.py  task=libero_90_data +setting_filter=LIVING_ROOM
# sbatch slurm/run_l40s.sbatch python scripts/fix_libero_data.py  task=libero_90_data +setting_filter=STUDY


# sbatch slurm/run_l40s.sbatch python scripts/fix_libero_data.py  task=libero_90_data +late_start=0
# sbatch slurm/run_l40s.sbatch python scripts/fix_libero_data.py  task=libero_90_data +late_start=10
# sbatch slurm/run_l40s.sbatch python scripts/fix_libero_data.py  task=libero_90_data +late_start=20
# sbatch slurm/run_l40s.sbatch python scripts/fix_libero_data.py  task=libero_90_data +late_start=30
# sbatch slurm/run_l40s.sbatch python scripts/fix_libero_data.py  task=libero_90_data +late_start=40
# sbatch slurm/run_l40s.sbatch python scripts/fix_libero_data.py  task=libero_90_data +late_start=50
# sbatch slurm/run_l40s.sbatch python scripts/fix_libero_data.py  task=libero_90_data +late_start=60
# sbatch slurm/run_l40s.sbatch python scripts/fix_libero_data.py  task=libero_90_data +late_start=70

for i in {0..9}; do
    echo python scripts/fix_libero_data.py  task=libero_10_data +task_nums=[${i}]
done



