#!/bin/bash
#SBATCH -JSlurmPythonExample                    # Job name
#SBATCH --account=gts-agarg35                   # charge account
#SBATCH -N1 --gres=gpu:RTX_6000:1               # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=12G                       # Memory per core
#SBATCH -t8:00:00                               # Duration of the job (8 hours)
#SBATCH -q embers                               # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=albertwilcox@gatech.edu     # E-mail address for notifications
#SBATCH --output=slurm_out/Report-%A.out
cd $HOME/p-agarg35-0/albert/quest        # Change to working directory

seeds=(0 1 2)
checkpoints="
multitask_model_epoch_0010.pth 
multitask_model_epoch_0020.pth 
multitask_model_epoch_0030.pth 
multitask_model_epoch_0040.pth 
multitask_model_epoch_0050.pth 
multitask_model_epoch_0060.pth 
multitask_model_epoch_0070.pth 
multitask_model_epoch_0080.pth 
multitask_model_epoch_0090.pth 
multitask_model_epoch_0100.pth 
multitask_model_epoch_0110.pth 
multitask_model_epoch_0120.pth 
multitask_model_epoch_0130.pth 
multitask_model_epoch_0140.pth 
multitask_model_epoch_0150.pth 
multitask_model_epoch_0160.pth 
multitask_model_epoch_0170.pth 
multitask_model_epoch_0180.pth 
multitask_model_epoch_0190.pth
multitask_model_final.pth
"
for checkpoint in ${checkpoints[@]}; do
    for seed in ${seeds[@]}; do
        python fix.py --config-name=train_prior.yaml \
            task=metaworld_ml45_prise \
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
            training.resume=true \
            checkpoint_path=experiments/metaworld/ML45_PRISE/diffusion_policy/dp_final/block_16/${seed}/stage_1/${checkpoint} \
            seed=${seed}

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
            training.resume=true \
            checkpoint_path=experiments/metaworld/ML45_PRISE/diffusion_policy/dp_final/block_16/${seed}/stage_2/${checkpoint} \
            seed=${seed}
    done
done



# python fix.py --config-name=train_prior.yaml \
#     task=metaworld_ml45_prise \
#     algo=diffusion_policy \
#     exp_name=dp_final \
#     variant_name=block_16 \
#     training.use_tqdm=false \
#     training.use_amp=true \
#     training.save_all_checkpoints=true \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.chunk_size=16 \
#     training.n_epochs=200 \
#     training.resume=true \
#     seed=1


# python fix.py --config-name=train_prior.yaml \
#     task=metaworld_ml45_prise \
#     algo=diffusion_policy \
#     exp_name=dp_final \
#     variant_name=block_16 \
#     training.use_tqdm=false \
#     training.use_amp=true \
#     training.save_all_checkpoints=true \
#     train_dataloader.persistent_workers=true \
#     train_dataloader.num_workers=6 \
#     make_unique_experiment_dir=false \
#     algo.chunk_size=16 \
#     training.n_epochs=200 \
#     training.resume=true \
#     seed=2

