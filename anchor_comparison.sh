# running
python main.py --device_num 1 --coco_path ./data/coco/ --output_dir VOC_out_resnet --wandb_run_name anchor_VOC_resnet --backbone resnet50 --eval_checkpoint_period 1
python main.py --device_num 0 --coco_path ./data/coco/ --output_dir VOC_out_SAM --wandb_run_name anchor_VOC_SAM --backbone SAM --eval_checkpoint_period 1
# to schedule
python main.py --device_num 2 --coco_path ./data/coco/ --output_dir VOC_out_SAM_neck --wandb_run_name anchor_VOC_SAM_only_neck --backbone SAM --eval_checkpoint_period 1 --only_neck
python main.py --device_num 3 --coco_path ./data/coco/ --output_dir VOC_out_resnet_UriahParams --wandb_run_name anchor_VOC_resnet_UriahParams --backbone resnet50 --num_query_position 50 --batch_size 4 --num_workers 4 --epochs 50000 --lr_drop 50000 --eval_checkpoint_period 5
# other sam params
python main.py --device_num 1 --coco_path ./data/coco/ --output_dir rerun_anchor_VOC_resnet_UriahParams --wandb_run_name rerun_anchor_VOC_resnet_UriahParams --backbone SAM --num_query_position 50 --batch_size 4 --num_workers 4 --epochs 50000 --lr_drop 50000 --eval_checkpoint_period 5
python main.py --device_num 2 --coco_path ./data/coco/ --output_dir rerun_anchor_VOC_resnet_UriahParams_regularLRdrop --wandb_run_name rerun_anchor_VOC_resnet_UriahParams_regularLRdrop --backbone SAM --num_query_position 50 --batch_size 4 --num_workers 4 --epochs 50000 --eval_checkpoint_period 5
python main.py --device_num 3 --coco_path ./data/coco/ --output_dir rerun_anchor_VOC_resnet_UriahParams_justNeck --wandb_run_name rerun_anchor_VOC_resnet_UriahParams_justNeck --backbone SAM --num_query_position 50 --batch_size 4 --num_workers 4 --epochs 50000 --lr_drop 50000 --eval_checkpoint_period 5 --only_neck
python main.py --device_num 4 --coco_path ./data/coco/ --output_dir rerun_anchor_VOC_resnet_UriahParams_frozenBackbone --wandb_run_name rerun_anchor_VOC_resnet_UriahParams_frozenBackbone --backbone SAM --num_query_position 50 --batch_size 4 --num_workers 4 --epochs 50000 --lr_drop 50000 --eval_checkpoint_period 5 --only_backbone
# higher BS
python main.py --device_num 5 --coco_path ./data/coco/ --output_dir rerun_anchor_VOC_SAM --wandb_run_name rerun_anchor_VOC_SAM --backbone SAM --eval_checkpoint_period 5
python main.py --device_num 6 --coco_path ./data/coco/ --output_dir rerun_anchor_VOC_SAM_only_neck --wandb_run_name rerun_anchor_VOC_SAM_only_neck --backbone SAM --eval_checkpoint_period 5 --only_neck
python main.py --device_num 7 --coco_path ./data/coco/ --output_dir rerun_anchor_VOC_SAM_only_backbone --wandb_run_name rerun_anchor_VOC_SAM_only_backbone --backbone SAM --eval_checkpoint_period 5 --only_backbone


