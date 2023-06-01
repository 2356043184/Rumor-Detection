OUTPUT_DIR=experiments/weibo/train_weibo_clip_bert_bilinear_more_channel
LOG=experiments/run_logs/train_weibo_clip_bert_bilinear_more_channel.log
nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port=29407 train.py \
--output_dir ${OUTPUT_DIR} \
--CUDA_VISIBLE_DEVICES 7 \
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
--batch_size 32 \
--batch_size_val 128 \
--image_size 224 \
--dataset weibo \
--attention_model bilinear \
--weight_decay 2e-5 \
--more_layer \
--exchange \
--loss_weight 1,1 \
--max_text_len 50 \
--num_workers 8 > ${LOG} 2>&1 &