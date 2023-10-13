python main.py --device_num 2 --coco_path ./data/coco/ --output_dir rerun_anchor_VOC_resnet_UriahParams_regularLRdrop \
--wandb_run_name rerun_anchor_VOC_resnet_UriahParams_regularLRdrop \
--backbone SAM --num_query_position 50 --batch_size 4 --num_workers 4 --epochs 50000 --eval_checkpoint_period 5 \
--coco_path /data/AllCellData/dataset/coco/cellpose \
--train_json train_RLE.json \
--val_json val_RLE.json \
--test_json test_RLE.json