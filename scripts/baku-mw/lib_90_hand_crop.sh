python train.py \
    --config-name=train_debug.yaml \
    task=metaworld_mt50_hybrid \
    algo=baku \
    algo/encoder=hybrid  \
    algo/aug=image \
    algo.chunk_size=10 \
    algo.encoder.backbone_type=clip \
    algo.encoder.finetune=false \
    algo.encoder.num_points=512 \
    +algo.policy.eecf=true \
    algo.encoder.do_crop=true \
    algo.encoder.do_hand_crop=true \
    +algo.encoder.tight_crop=true \
    ~task.shape_meta.observation.lowdim \
    +task.shape_meta.observation.lowdim={ee_open:1} \
    training.save_interval=20 \
    train_dataloader.num_workers=8 \
    rollout.interval=20 \
    task.demos_per_env=50 \
    training.n_epochs=101 


    
exp_name="final"
blocks=(10)
seeds=(0 1)
backbones=("resnet18" "clip")
finetunes=(true false)
downsamples=(512)

for block in ${blocks[@]}; do
    for seed in ${seeds[@]}; do
        for backbone in ${backbones[@]}; do
            for finetune in ${finetunes[@]}; do
                for downsample in ${downsamples[@]}; do
                    # ds to 1024
                    sbatch slurm/run_coe.sbatch python train.py \
                        --config-name=train_prior.yaml \
                        task=metaworld_mt50_hybrid \
                        exp_name=${exp_name} \
                        variant_name=tight_hand_crop_ds_${downsample}_${backbone}_ft_${finetune}_block_${block} \
                        algo=baku \
                        algo/encoder=hybrid  \
                        algo/aug=image \
                        algo.chunk_size=${block} \
                        algo.encoder.backbone_type=${backbone} \
                        algo.encoder.finetune=${finetune} \
                        algo.encoder.num_points=${downsample} \
                        +algo.policy.eecf=true \
                        algo.encoder.do_crop=true \
                        algo.encoder.do_hand_crop=true \
                        +algo.encoder.tight_crop=true \
                        ~task.shape_meta.observation.lowdim \
                        +task.shape_meta.observation.lowdim={ee_open:1} \
                        training.save_interval=20 \
                        train_dataloader.num_workers=8 \
                        rollout.interval=20 \
                        task.demos_per_env=50 \
                        training.n_epochs=101 \
                        seed=${seed}
                done
            done
        done
    done
done


