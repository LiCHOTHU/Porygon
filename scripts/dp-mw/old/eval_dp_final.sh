



sbatch slurm/run_rtx6000.sbatch python fix.py \
    task=metaworld_ml45_prise_fewshot \
    algo=diffusion_policy \
    exp_name=dp_final \
    variant_name=block_16 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.chunk_size=16 \
    training.n_epochs=200 \
    training.auto_continue=true \
    seed=0



sbatch slurm/run_rtx6000.sbatch python fix.py \
    task=metaworld_ml45_prise_fewshot \
    algo=diffusion_policy \
    exp_name=dp_final \
    variant_name=block_16 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.chunk_size=16 \
    training.n_epochs=200 \
    training.auto_continue=true \
    seed=1


sbatch slurm/run_rtx6000.sbatch python fix.py \
    task=metaworld_ml45_prise_fewshot \
    algo=diffusion_policy \
    exp_name=dp_final \
    variant_name=block_16 \
    training.use_tqdm=false \
    training.use_amp=true \
    training.save_all_checkpoints=true \
    train_dataloader.persistent_workers=true \
    train_dataloader.num_workers=6 \
    make_unique_experiment_dir=false \
    algo.chunk_size=16 \
    training.n_epochs=200 \
    training.auto_continue=true \
    seed=2

