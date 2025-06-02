task="libero_90_hybrid"
exp_name="head_sweep"
algos=(
    "dit_head"
    "dit_head_2"
)
hidden_dims=(256 512)
num_layerss=(4 8)

for algo in ${algos[@]}; do
    for hidden_dim in ${hidden_dims[@]}; do
        for num_layers in ${num_layerss[@]}; do
            echo python train.py \
                --config-name=train_prior.yaml \
                task=${task} \
                exp_name=${exp_name} \
                variant_name=${algo}_hd_${hidden_dim}_nl_${num_layers} \
                pace_copy=true \
                algo=${algo} \
                algo.chunk_size=15 \
                algo.policy.hidden_dim=${hidden_dim} \
                algo.policy.num_layers=${num_layers} \
                training.save_interval=10 \
                train_dataloader.num_workers=4 \
                rollout.interval=10 \
                task.demos_per_env=50 \
                training.n_epochs=101 \
                training.use_amp=true
        done
    done
done

algo="da_style_head"
embedding_dims=(240 480)
num_layerss=(4 8)
attn_per_layers=(1)
for embedding_dim in ${embedding_dims[@]}; do
    for attn_per_layer in ${attn_per_layers[@]}; do
        for num_layers in ${num_layerss[@]}; do
            echo python train.py \
                --config-name=train_prior.yaml \
                task=${task} \
                exp_name=${exp_name} \
                variant_name=${algo}_ed_${embedding_dim}_nl_${num_layers}_apl_${attn_per_layer} \
                pace_copy=true \
                algo=${algo} \
                algo.chunk_size=15 \
                algo.policy.embedding_dim=${embedding_dim} \
                algo.policy.num_layers=${num_layers} \
                algo.policy.num_sa_layers=${attn_per_layer} \
                algo.policy.num_xa_layers=${attn_per_layer} \
                training.save_interval=10 \
                train_dataloader.num_workers=4 \
                rollout.interval=10 \
                task.demos_per_env=50 \
                training.n_epochs=101 \
                training.use_amp=true
        done
    done
done



