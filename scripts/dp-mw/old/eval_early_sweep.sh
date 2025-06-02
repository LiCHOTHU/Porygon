task="metaworld_ml45_prise"
algo="diffusion_policy"
exp_name="dp_final"
stage=1
variant_name="block_16"
checkpoints="
multitask_model_epoch_0050.pth
multitask_model_epoch_0100.pth
multitask_model_epoch_0150.pth
multitask_model_final.pth
"
seeds=(3 4)




for seed in ${seeds[@]}; do
    for checkpoint in ${checkpoints[@]}; do
        sbatch slurm/run_rtx6000.sbatch python evaluate.py \
            task=$task \
            algo=$algo \
            exp_name=${exp_name}_early_sweep \
            variant_name=${variant_name}_${checkpoint} \
            stage=$stage \
            training.use_tqdm=false \
            seed=$seed \
            checkpoint_path=/storage/home/hcoda1/1/awilcox31/p-agarg35-0/albert/quest/experiments/metaworld/ML45_PRISE/diffusion_policy/${exp_name}/${variant_name}/${seed}/stage_${stage}/${checkpoint}
    done
done

