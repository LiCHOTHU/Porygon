algo="diffusion_policy"


python scripts/make_bar_plot.py \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_clip_ft_false_block_8_multitask_model_epoch_0020.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_clip_ft_false_block_8_multitask_model_epoch_0040.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_clip_ft_false_block_8_multitask_model_epoch_0060.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_clip_ft_false_block_8_multitask_model_epoch_0080.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_clip_ft_false_block_8_multitask_model_epoch_0100.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_clip_ft_true_block_8_multitask_model_epoch_0020.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_clip_ft_true_block_8_multitask_model_epoch_0040.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_clip_ft_true_block_8_multitask_model_epoch_0060.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_clip_ft_true_block_8_multitask_model_epoch_0080.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_clip_ft_true_block_8_multitask_model_epoch_0100.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_resnet18_ft_false_block_8_multitask_model_epoch_0020.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_resnet18_ft_false_block_8_multitask_model_epoch_0040.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_resnet18_ft_false_block_8_multitask_model_epoch_0060.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_resnet18_ft_false_block_8_multitask_model_epoch_0080.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_resnet18_ft_false_block_8_multitask_model_epoch_0100.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_resnet18_ft_true_block_8_multitask_model_epoch_0020.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_resnet18_ft_true_block_8_multitask_model_epoch_0040.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_resnet18_ft_true_block_8_multitask_model_epoch_0060.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_resnet18_ft_true_block_8_multitask_model_epoch_0080.pth \
        experiments/evaluate/libero/libero_90/${algo}/final_ckpt_sweep/tight_hand_crop_ds_512_resnet18_ft_true_block_8_multitask_model_epoch_0100.pth \
    --labels  \
        clip_ft_false_epoch_020 \
        clip_ft_false_epoch_040 \
        clip_ft_false_epoch_060 \
        clip_ft_false_epoch_080 \
        clip_ft_false_epoch_100 \
        clip_ft_true_epoch_020 \
        clip_ft_true_epoch_040 \
        clip_ft_true_epoch_060 \
        clip_ft_true_epoch_080 \
        clip_ft_true_epoch_100 \
        resnet18_ft_false_epoch_020 \
        resnet18_ft_false_epoch_040 \
        resnet18_ft_false_epoch_060 \
        resnet18_ft_false_epoch_080 \
        resnet18_ft_false_epoch_100 \
        resnet18_ft_true_epoch_020 \
        resnet18_ft_true_epoch_040 \
        resnet18_ft_true_epoch_060 \
        resnet18_ft_true_epoch_080 \
        resnet18_ft_true_epoch_100 \
    --fname ${algo}-upce-final --xtick-rotation 90 --font-size-annotation 8 --font-size-xticks 10
