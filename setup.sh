# TODO: generate annotations
#python voc2coco.py --split train
#python voc2coco.py --split val
#python voc2coco.py --split test

#mkdir data/coco/annotations
#mkdir data/coco/train
#mkdir data/coco/val

#cp -r data/VOCdevkit/VOC2007/JPEGImages/* data/coco/train/
#cp -r data/VOCdevkit/VOC2007/JPEGImages/* data/coco/val/

# sam stuff
git clone https://github.com/damaggu/segment-anything.git
mkdir pretrained_models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P pretrained_models/

mv segment-anything segment_anything
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/segment_anything
export PYTHONPATH=$PYTHONPATH:$(pwd)/segment_anything/segment_anything