# running
python main.py --device_num 1 --coco_path ./data/coco/ --output_dir VOC_out_resnet --wandb_run_name anchor_VOC_resnet --backbone resnet50 --eval_checkpoint_period 1
python main.py --device_num 0 --coco_path ./data/coco/ --output_dir VOC_out_SAM --wandb_run_name anchor_VOC_SAM --backbone SAM --eval_checkpoint_period 1
# to schedule
python main.py --device_num 0 --coco_path ./data/coco/ --output_dir VOC_out_SAM_neck --wandb_run_name anchor_VOC_SAM_only_neck --backbone SAM --eval_checkpoint_period 1 --only_neck
python main.py --device_num 1 --coco_path ./data/coco/ --output_dir VOC_out_resnet_UriahParams --wandb_run_name anchor_VOC_resnet_UriahParams --backbone resnet50 --num_query_position 50 --batch_size 4 --num_workers 4 --epochs 50000 --lr_drop 50000 --eval_checkpoint_period 5