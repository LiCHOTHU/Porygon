
# python fix.py --config-name=train_fewshot.yaml \
#     task=metaworld_ml45_prise_fewshot \
#     algo=diffusion_policy \
#     training.use_amp=true \
#     training.save_all_checkpoints=true \
#     train_dataloader.num_workers=6 \
#     algo.chunk_size=16 \
#     training.n_epochs=200 \
#     checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/diffusion_policy/dp_final/block_16/0/stage_1/multitask_model_final.pth \
#     task.env_runner.debug=true \
#     logging.mode=disabled




python fix.py --config-name=train_fewshot.yaml \
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



python fix.py --config-name=train_fewshot.yaml \
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


python fix.py --config-name=train_fewshot.yaml \
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

