# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from AnchorDETR.datasets.torchvision_datasets import CocoDetection as TvCocoDetection
from AnchorDETR.util.misc import get_local_rank, get_local_size
import AnchorDETR.datasets.transforms as T


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1, nuclear=False):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.nuclear = nuclear

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if self.nuclear:
            img = img[2,:,:]
            img = img.unsqueeze(0)
            # repeat
            img = img.repeat(3,1,1)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, args):
    # #############################################################
    # # coco augments
    # normalize = T.Compose([
    #     T.ToTensor(),
    #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    SAM_MEAN = [123.675, 116.28, 103.53]
    SAM_STD = [58.395, 57.12, 57.375]

    SAM_MEAN = [x / 255.0 for x in SAM_MEAN]
    SAM_STD = [x / 255.0 for x in SAM_STD]

    normalize = T.Compose([
        T.ToTensor(),
        # T.Normalize(SAM_MEAN, SAM_STD),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    more_scales = [280, 360, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 900, 1000, 1100, 1200]
    angles = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5,
              135.0, 157.5, 180.0, 202.5, 225.0, 247.5,
              270.0, 292.5, 315.0, 337.5]

    if image_set == 'train':
        if args.additional_augmentations == 'none':
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        # T.RandomResize(scales, max_size=1333),
                    ])
                ),
                #Resize everything to 1024 for SAM testing, #TODO: remove me later?
                T.FixedResize([1024, 1024]),
                normalize,
            ])
        elif args.additional_augmentations == 'more_scales':
            if image_set == 'train':
                return T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomSelect(
                        T.RandomResize(scales, max_size=1333),
                        T.Compose([
                            T.RandomResize(more_scales),
                            T.RandomSizeCrop(250, 1100),
                        ])
                    ),
                    # Resize everything to 1024 for SAM testing, #TODO: remove me later?
                    T.FixedResize([1024, 1024]),
                    T.RandomColorAugmentation(),
                    normalize,
                ])

        elif args.additional_augmentations == 'newest':
            if image_set == 'train':
                return T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomSelect(
                        T.RandomResize(scales, max_size=1333),
                        T.Compose([
                            T.RandomResize(more_scales),
                            T.RandomSizeCrop(250, 1100),
                        ])
                    ),
                    # Resize everything to 1024 for SAM testing, #TODO: remove me later?
                    T.FixedResize([1024, 1024]),
                    T.RandomColorAugmentation(),
                    normalize,
                ])

        elif args.additional_augmentations == 'random_rotation_and_flip':

            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotate(angles),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        # T.RandomResize(scales, max_size=1333),
                    ])
                ),
                # Resize everything to 1024 for SAM testing, #TODO: remove me later?
                T.FixedResize([1024, 1024]),
                normalize,
            ])
        elif args.additional_augmentations == 'qilin':
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.FixedResize([512, 512]),  # directly return 512x512 image
                    T.RandomSelect(
                        # zoom in to max 256x256
                        T.Compose([
                            T.RandomSizeCrop(256, 512),
                            T.FixedResize([512, 512]),
                        ]),
                        # zoom out to max 1024x1024
                        T.Compose([
                            T.RandomPad(512),
                            T.RandomSizeCrop(400, 1024),
                            T.FixedResize([512, 512]),
                        ]), p=0.5
                    ), p=0.33
                ),
                T.FixedResize([1024, 1024]),
                normalize,
            ])
        elif args.additional_augmentations == 'combined':
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomResize(scales, max_size=1333),
                T.RandomRotate(angles),
                T.RandomSelect(
                    T.FixedResize([512, 512]),  # directly return 512x512 image
                    T.RandomSelect(
                        # zoom in to max 256x256
                        T.Compose([
                            T.RandomSizeCrop(256, 512),
                            T.FixedResize([512, 512]),
                        ]),
                        # zoom out to max 1024x1024
                        T.Compose([
                            T.RandomPad(512),
                            T.RandomSizeCrop(400, 1024),
                            T.FixedResize([512, 512]),
                        ]), p=0.5
                    ), p=0.33
                ),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        # T.RandomResize(scales, max_size=1333),
                    ])
                ),
                T.FixedResize([1024, 1024]),
                normalize,
            ])

        elif args.additional_augmentations == 'combined_v2':
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        # T.RandomResize(scales, max_size=1333),
                    ])
                ),
                T.RandomRotate(angles),
                T.FixedResize([1024, 1024]),
                normalize,
            ])
        else:
            raise ValueError(f'unknown {args.additional_augmentations}')

    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            # T.RandomResize([800], max_size=1333),
            #Resize everything to 1024 for SAM testing
            T.FixedResize([1024,1024]),
            # T.RandomRotate(angles),
            normalize,
        ])
    # #############################################################

    #############################################################
    # cell image augs. naive versioon: assume same 256x256 size for all input image files.
    # normalize = T.Compose([
    #     T.ToTensor(),
    #     T.Normalize([0.3, 0.3, 0.3], [0.3, 0.3, 0.3])
    # ])
    #
    # scales = [256] # fixed input size 256x256
    #
    # if image_set == 'train':
    #     return T.Compose([
    #         T.RandomHorizontalFlip(),
    #         T.RandomSelect(
    #             T.FixedResize([256,256]), # directly return 256x256 image
    #             T.RandomSelect(
    #                 # zoom in to max 128x128
    #                 T.Compose([
    #                     T.RandomSizeCrop(128, 256),
    #                     T.FixedResize([256,256]),
    #                 ]),
    #                 # zoom out to max 512x512
    #                 T.Compose([
    #                     T.RandomPad(256),
    #                     T.RandomSizeCrop(200, 512),
    #                     T.FixedResize([256,256]),
    #                 ]), p=0.5
    #             ), p=0.33
    #         ),
    #
    #         # # zoom in to max 128x128
    #         # T.Compose([
    #         #     T.RandomSizeCrop(128, 256),
    #         #     T.FixedResize([256,256]),
    #         # ]),
    #
    #         # # zoom out to max 512x512
    #         # T.Compose([
    #         #             T.RandomPad(256),
    #         #             T.RandomSizeCrop(200, 512),
    #         #             T.FixedResize([256,256]),
    #         #         ]),
    #         normalize,
    #     ])
    #
    # if image_set == 'val' or image_set == 'test': # always use 256x256 image so no aug required
    #     return T.Compose([
    #         T.RandomResize([256], max_size=1333),
    #         normalize,
    #     ])
    #############################################################

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    # PATHS = {
    #     "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
    #     "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    #     "test": (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json'),
    # }

    # voc2007 paths
    PATHS = {
        "train": (root / args.train_path, root / args.json_root / args.train_json),
        "val": (root / args.val_path, root / args.json_root / args.val_json),
        "test": (root / args.test_path, root / args.json_root / args.test_json),
    }

    # omnipose paths
    # PATHS = {
    #     "train": (root / "train_256", root / "annotations" / f'train_BF_RLE_256.json'),
    #     "val": (root / "test_256", root / "annotations" / f'test_BF_RLE_256.json'),
    # }

    # # tissuenet paths
    # PATHS = {
    #     "train": (root / "train_crop_256", root / "annotations" / f'train_crop_256_RLE_WholeCell2.json'),
    #     "val": (root / "val", root / "annotations" / f'val_RLE_WholeCell2.json'),
    # }


    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set, args), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), nuclear=args.delete_nuclear_channel)
    return dataset


if __name__ == '__main__':

    # make args as namespace
    class Args:
        def __init__(self):
            self.additional_augmentations = 'more_scales'

    args = Args()

    transforms  = make_coco_transforms('train', args)

    from skimage import data
    # test it
    example_image = data.immunohistochemistry()
    # to PIL image
    from PIL import Image
    example_image = Image.fromarray(example_image)
    import matplotlib.pyplot as plt

    plt.imshow(example_image)
    plt.show()

    targets = [{'segmentation': [[1, 1, 1, 2, 2, 2, 2, 1]], 'bbox': [1, 1, 1, 1], 'category_id': 1}]


    # transforms
    example_image = transforms(example_image, {'image_id': 0, 'annotations': targets})

    plt.imshow(example_image[0].permute(1,2,0))
    plt.show()