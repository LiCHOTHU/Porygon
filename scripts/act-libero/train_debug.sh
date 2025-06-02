# python train.py \
#     --config-name=train_debug.yaml \
#     task=libero_10_hybrid \
#     algo=baku \
#     algo/encoder=hybrid  \
#     algo.chunk_size=10 \
#     algo.encoder.backbone_type=clip \
#     algo.encoder.finetune=false \
#     algo.encoder.num_points=1024 \
#     +algo.policy.eecf=true \
#     algo/aug=color_jitter \
#     ~task.shape_meta.observation.lowdim \
#     +task.shape_meta.observation.lowdim={robot0_gripper_qpos:2} \
#     rollout.interval=20 \
#     task.demos_per_env=50 

exp_name="final"
blocks=(10)
# seeds=(0 1)
seeds=(2 3 4)
backbones=("clip")
finetunes=(false)
downsamples=(512)


python train.py \
    --config-name=train_debug.yaml \
    task=libero_10_hybrid \
    algo=act \
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
    training.save_interval=10 \
    train_dataloader.num_workers=4 \
    rollout.interval=100 \
    task.demos_per_env=50 \
    training.n_epochs=101


