OUTPUT_DIR=output/exp7
nohup python -m torch.distributed.launch --nproc_per_node=1 train.py \
--output_dir ${OUTPUT_DIR} \
--CUDA_VISIBLE_DEVICES 0 \
--do_train \
--n_epochs 30 \
--lr 3e-5 \
--image_model_type resnet50 \
--language_model_type bert \
--pretrained_image \
--freeze_language \
--pretrained_language \
--dataset pheme \
--batch_size 128 \
--batch_size_val 128 \
--max_text_len 50 \
--image_size 224 \
--expand_language \
--expand_image \
--attention_model \
--exchange \
--num_workers 8 > ${OUTPUT_DIR}/run.log 2>&1 &