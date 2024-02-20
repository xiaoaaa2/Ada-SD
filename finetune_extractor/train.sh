export PATH="/home/anaconda3/envs/voice2/bin:$PATH"
which python

mkdir -p logs
mkdir -p saved_models

dataset='cn12'
tag='_exp1'

python train.py \
    --batch_size 32 \
    --num_classes_in_batch 4 \
    --model_type saved_models/wav2vec_small.pt \
    --lr 1e-5 \
    --lr_scheduler 1 \
    --pct_warmup_steps 0.1 \
    --num_freeze_steps 0 \
    --epochs 20 \
    --logfile logs/${dataset}${tag}.log \
    --seed 42 \
    --optimizer_type Adam \
    --grad_acc_step 2 \
    --custom_embed_size 256 \
    --with_relu 0 \
    --save_path saved_models/model_${dataset}${tag}.pt \
    --resume_training 1 \
    --data_type cn_1 \
    --dropout_val 0 \
    --refine_matrix 0 \
    --g_blur 0.0 \
    --p_pct 90 \
    --mse_fac 0.5 \
    --margin 0.0
