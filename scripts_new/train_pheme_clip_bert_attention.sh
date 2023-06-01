OUTPUT_DIR=experiments/pheme/train_pheme_clip_bert_attention
LOG=experiments/run_logs/train_pheme_clip_bert_attention.log
nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=29401 train.py \
--output_dir ${OUTPUT_DIR} \
--CUDA_VISIBLE_DEVICES 1 \
--do_train \
--n_epochs 30 \
--lr 1.5e-4 \
--image_model_type clip \
--language_model_type bert \
--pretrained_image \
--pretrained_language \
--freeze_image \
--freeze_language \
--expand_image \
--expand_language \
--batch_size 64 \
--batch_size_val 128 \
--image_size 224 \
--dataset pheme \
--attention_model simple \
--weight_decay 2e-5 \
--loss_weight 1,2 \
--max_text_len 50 \
--num_workers 8 > ${LOG} 2>&1 &