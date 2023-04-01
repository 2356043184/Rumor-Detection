OUTPUT_DIR=output/exp1
nohup python -m torch.distributed.launch --nproc_per_node=1 train.py \
--output_dir ${OUTPUT_DIR} \
--CUDA_VISIBLE_DEVICES 0 \
--do_train \
--n_epochs 15 \
--lr 3e-5 \
--image_model_type resnet50 \
--language_model_type transformer \
--pretrained_image \
--csv_path datasets/content_noid.csv \
--image_folder datasets/pheme_images_jpg \
--train_id_file datasets/train_ids.txt \
--test_id_file datasets/test_ids.txt \
--batch_size 128 \
--batch_size_val 128 \
--max_text_len 40 \
--image_size 224 \
--num_workers 8 > ${OUTPUT_DIR}/run.log 2>&1 &




