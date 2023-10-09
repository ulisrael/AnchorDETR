import os
import sys
import time
import math
import torch
import argparse

import torch.nn as nn
import torch.nn.functional as F

from typing import List

from models.transformer import build_transformer
from models.matcher import build_matcher
from models.anchor_detr import SetCriterion, PostProcess

## Debug imports
# from transformers import DetrImageProcessor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import util.misc as utils
from util.misc import nested_tensor_from_tensor_list, NestedTensor



def get_args_parser():
    parser = argparse.ArgumentParser('AnchorDETR Detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=[], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--num_feature_levels', default=1, type=int, help='number of feature levels')
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0., type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_query_position', default=300, type=int,
                        help="Number of query positions")
    parser.add_argument('--num_query_pattern', default=3, type=int,
                        help="Number of query patterns")
    parser.add_argument('--spatial_prior', default='learned', choices=['learned', 'grid'],
                        type=str,help="Number of query patterns")
    parser.add_argument('--attention_type',
                        # default='nn.MultiheadAttention',
                        default="RCDA",
                        choices=['RCDA', 'nn.MultiheadAttention'],
                        type=str,help="Type of attention module")
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--eval_set', default='val', choices=['val', 'test'],
                        type=str,help="dataset to evaluate")
    parser.add_argument('--coco_path', default='/data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='/data/detr-workdir/r50-dc5',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', default=False, action='store_true', help='whether to resume from last checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser.parse_args()

import torchvision



class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, detr_processor, split='train'):
        # if split == 'train':
        #     ann_file = os.path.join('./data/coco', "voc_coco_train.json")
        #     img_folder = os.path.join('./data/VOC2007/JPEGImages')
        # elif split == 'val':
        #     ann_file = os.path.join('./data/coco', "voc_coco_val.json")
        # elif split == 'test':
        #     ann_file = os.path.join('./data/coco', "voc_coco_test.json")
        # else:
        #     raise ValueError(f"Invalid split {split}")
        ann_file = os.path.join(img_folder, 'annotations/train.json')
        img_folder = os.path.join(img_folder, 'train')
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = detr_processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]

    batch_out = {}
    batch_out['pixel_values'] = encoding['pixel_values']
    batch_out['pixel_mask'] = encoding['pixel_mask']
    batch_out['labels'] = labels
    return batch_out


def viz_coco_img(data_dir, train_ds, img, tgt):
    import numpy as np
    import os
    from PIL import Image, ImageDraw
    import torchvision.transforms as T

    fig, ax = plt.subplots(1,2, figsize=(10,5))

    # based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
    # image_ids = train_ds.coco.getImgIds()
    # # let's pick a random image
    # image_id = image_ids[np.random.randint(0, len(image_ids))]
    # print('Image n°{}'.format(image_id))
    orig_id = int(tgt['image_id'])
    orig_image = train_ds.coco.loadImgs(orig_id)[0]
    orig_image = Image.open(os.path.join(data_dir,'train', orig_image['file_name']))

    draw = ImageDraw.Draw(orig_image, "RGBA")

    annotations = train_ds.coco.imgToAnns[orig_id]

    cats = train_ds.coco.cats
    id2label = {k: v['name'] for k, v in cats.items()}

    for annotation in annotations:
        box = annotation['bbox']
        class_idx = annotation['category_id']
        x, y, w, h = tuple(box)
        draw.rectangle((x, y, x + w, y + h), outline='red', width=5)
        draw.text((x, y), id2label[class_idx], fill='white')

    ax[0].imshow(orig_image)


    image = T.ToPILImage()(img)
    draw = ImageDraw.Draw(image, "RGBA")

    for i in range(len(tgt['boxes'])):
        box = tgt['boxes'][i]
        class_idx = int(tgt['class_labels'][i])
        ih, iw = tgt['size']
        # x_min, y_min, x_max, y_max = tuple(box)
        # x_min, y_min, x_max, y_max = x_min * w, y_min * h, x_max * w, y_max * h
        # draw.rectangle((x_min, y_min, x_max, y_max), outline='red', width=1)
        # draw.text((x_min, y_min), id2label[class_idx], fill='white')
        xc, yc, w, h = tuple(box)
        x, y, w, h = (xc - w/2.) * iw, (yc - h/2) * ih, w * iw, h * ih
        draw.rectangle((x, y, x + w, y + h), outline='red', width=15)
        draw.text((x, y), id2label[class_idx], fill='white')

    ax[1].imshow(image)

    return fig

class SAMAnchorDETR(nn.Module):
    """ This is the AnchorDETR module that performs object detection """

    def __init__(self, transformer,  num_feature_levels=1, aux_loss=True, in_channels=2048):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model

        #TODO NEED this info from backbone for imp - These are hard coded in
        # len(backbone.strides)
        # backbone.num_channels

        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            # num_backbone_outs = len(backbone.strides)
            num_backbone_outs = 1 # this is hard code
            input_proj_list = []
            for _ in range(num_backbone_outs):
                # in_channels = backbone.num_channels
                if _ == 0:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                else:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    # nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.aux_loss = aux_loss

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, feats: List[List]):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        srcs = []
        masks = []
        for l, feat in enumerate(feats):
            src, mask = feat[0], feat[1]
            srcs.append(self.input_proj[l](src).unsqueeze(1))
            masks.append(mask)
            assert mask is not None

        srcs = torch.cat(srcs, dim=1)

        outputs_class, outputs_coord = self.transformer(srcs, masks)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    def forward_inference(self, feat_list, targets):
        # TODO we need orig img size for postprocessing
        pass

        # outputs = self.forward(feat_list)
        #
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = self.postprocessors['bbox'](outputs, orig_target_sizes)
        # res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        #
        # return res

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def build_samanchor(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    if args.num_classes is None:
        args.num_classes = 20 if args.dataset_file != 'coco' else 91
        if args.dataset_file == "coco_panoptic":
            # for panoptic, we just add a num_classes that is large enough to hold
            # max_obj_id + 1, but the exact value doesn't really matter
            args.num_classes = 250

    device = torch.device(args.device)
    # device = torch.device('cpu')

    transformer = build_transformer(args)

    model = SAMAnchorDETR(
        transformer,
        num_feature_levels=1,
        aux_loss=True,
        # in_channels=args.in_channels,
    )

    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(args.num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors


# SAM test anchor
class TestAnchorDETR(nn.Module):
    def __init__(self, backbone, model):
        super().__init__()
        self.backbone = backbone
        self.model = model

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features = self.backbone(samples)


        feat_list = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            feat_list.append([src, mask])

        out = self.model(feat_list)

        return out



if __name__ == "__main__":
    args = get_args_parser()
    print(args)

    bs = args.batch_size

    device = torch.device(args.device)
    # device = torch.device('cpu')

    # output dir
    output_dir = args.output_dir

    # define dataset and preprocessing
    data_dir = './data/coco'
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    train_ds = CocoDetection(data_dir, processor, split='train')
    train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=bs, shuffle=True)

    # define backbone
    backbone_ = torchvision.models.resnet50(pretrained=True)
    layers = list(backbone_.children())[:-2]
    backbone = nn.Sequential(*layers)  # shape [batch, 2048, h, w]

    # define model
    model_without_ddp = SAMAnchorDETR(args)

    ## Print utils for testing
    # x, y = train_ds[0]
    # test_img = viz_coco_img(data_dir, train_ds, x, y)
    # test_img.show()

    # define optimizer
    max_norm = args.clip_max_norm
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    # setup stuff for training
    backbone.to(device)
    model_without_ddp.to(device)
    # model_without_ddp.train_step(feat_list, labels)
    model_without_ddp.train()
    model_without_ddp.criterion.train()


    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        print()
        print(f'starting epoch {epoch} out of {args.epochs}')

        for batch in train_dataloader:
            pass

            imgs, masks, labels = batch['pixel_values'].to(device), batch['pixel_mask'].to(device), batch['labels']

            for label in labels:
                label['labels'] = label.pop('class_labels')

            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]



            features = backbone(imgs)
            masks = F.interpolate(masks[None].float(), size=features.shape[-2:]).to(torch.bool)[0]

            # invert mask bc anchor deter is 1 for invalid pixels and 0 for valid pixels
            # different from hugging face DETR preprocessor which is 1 for valid pixels and 0 for invalid pixels
            masks = ~masks

            feat_list = [[features, masks]]

            outputs = model_without_ddp(feat_list)
            loss_dict = model_without_ddp.criterion(outputs, labels)
            weight_dict = model_without_ddp.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = loss_dict
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()

            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model_without_ddp.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model_without_ddp.parameters(), max_norm)
            optimizer.step()
            lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [os.path.join(output_dir, 'checkpoint.pth')]
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                new_path = os.path.join(output_dir, f'checkpoint{epoch:04}.pth')
                checkpoint_paths.append(new_path)
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        print(f'loss at the end of epoch {epoch}: {losses}')


"""
OLD CODE that works
"""

## old sam model ##

# class SAMAnchorDETR(nn.Module):
#     """ This is the AnchorDETR module that performs object detection """
#
#     def __init__(self, transformer,  num_feature_levels=1, aux_loss=True):
#         """ Initializes the model.
#         Parameters:
#             transformer: torch module of the transformer architecture. See transformer.py
#             num_classes: number of object classes
#             aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
#         """
#         super().__init__()
#         self.transformer = transformer
#         hidden_dim = transformer.d_model
#
#         #TODO NEED this info from backbone for imp - These are hard coded in
#         # len(backbone.strides)
#         # backbone.num_channels
#
#         self.num_feature_levels = num_feature_levels
#         if num_feature_levels > 1:
#             # num_backbone_outs = len(backbone.strides)
#             num_backbone_outs = 1 # this is hard code
#             input_proj_list = []
#             for _ in range(num_backbone_outs):
#                 # in_channels = backbone.num_channels
#                 in_channels = 2048
#                 if _ == 0:
#                     input_proj_list.append(nn.Sequential(
#                         nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
#                         nn.GroupNorm(32, hidden_dim),
#                     ))
#                 else:
#                     input_proj_list.append(nn.Sequential(
#                         nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
#                         nn.GroupNorm(32, hidden_dim),
#                     ))
#             self.input_proj = nn.ModuleList(input_proj_list)
#         else:
#             self.input_proj = nn.ModuleList([
#                 nn.Sequential(
#                     # nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
#                     nn.Conv2d(2048, hidden_dim, kernel_size=1),
#                     nn.GroupNorm(32, hidden_dim),
#                 )])
#         self.aux_loss = aux_loss
#
#         for proj in self.input_proj:
#             nn.init.xavier_uniform_(proj[0].weight, gain=1)
#             nn.init.constant_(proj[0].bias, 0)
#
#     def forward(self, feats: List[List]):
#         """ The forward expects a NestedTensor, which consists of:
#                - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
#                - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
#
#             It returns a dict with the following elements:
#                - "pred_logits": the classification logits (including no-object) for all queries.
#                                 Shape= [batch_size x num_queries x (num_classes + 1)]
#                - "pred_boxes": The normalized boxes coordinates for all queries, represented as
#                                (center_x, center_y, height, width). These values are normalized in [0, 1],
#                                relative to the size of each individual image (disregarding possible padding).
#                                See PostProcess for information on how to retrieve the unnormalized bounding box.
#                - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
#                                 dictionnaries containing the two above keys for each decoder layer.
#         """
#
#         srcs = []
#         masks = []
#         for l, feat in enumerate(feats):
#             src, mask = feat[0], feat[1]
#             srcs.append(self.input_proj[l](src).unsqueeze(1))
#             masks.append(mask)
#             assert mask is not None
#
#         srcs = torch.cat(srcs, dim=1)
#
#         outputs_class, outputs_coord = self.transformer(srcs, masks)
#
#         out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
#         if self.aux_loss:
#             out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
#
#         return out
#
#     def forward_inference(self, feat_list, targets):
#         # TODO we need orig img size for postprocessing
#         pass
#
#         # outputs = self.forward(feat_list)
#         #
#         # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#         # results = self.postprocessors['bbox'](outputs, orig_target_sizes)
#         # res = {target['image_id'].item(): output for target, output in zip(targets, results)}
#         #
#         # return res
#
#     @torch.jit.unused
#     def _set_aux_loss(self, outputs_class, outputs_coord):
#         # this is a workaround to make torchscript happy, as torchscript
#         # doesn't support dictionary with non-homogeneous values, such
#         # as a dict having both a Tensor and a list.
#         return [{'pred_logits': a, 'pred_boxes': b}
#                 for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


## old test anchor wrapper ##
# ### test anchor that works by itself - changing for SAM
# class TestAnchorDETR_old(nn.Module):
#     def __init__(self, backbone, model):
#         super().__init__()
#         self.backbone = backbone
#         self.model = model
#
#
#
#     def forward(self, samples: NestedTensor):
#         if not isinstance(samples, NestedTensor):
#             samples = nested_tensor_from_tensor_list(samples)
#         features = self.backbone(samples)
#
#
#         feat_list = []
#         for l, feat in enumerate(features):
#             src, mask = feat.decompose()
#             feat_list.append([src, mask])
#
#         out = self.model(feat_list)
#
#         return out