# TODO: generate annotations
#python voc2coco.py --split train
#python voc2coco.py --split val
#python voc2coco.py --split test

mkdir data/coco/annotations
mkdir data/coco/train
mkdir data/coco/val

cp -r data/VOCdevkit/VOC2007/JPEGImages/* data/coco/train/
cp -r data/VOCdevkit/VOC2007/JPEGImages/* data/coco/val/