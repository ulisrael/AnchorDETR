# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

# evaluation figure drawing
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image, ImageDraw
from skimage.io import imread
import wandb


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
                        type=str, help="Number of query patterns")
    parser.add_argument('--attention_type',
                        # default='nn.MultiheadAttention',
                        default="RCDA",
                        choices=['RCDA', 'nn.MultiheadAttention'],
                        type=str, help="Type of attention module")
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
                        type=str, help="dataset to evaluate")
    parser.add_argument('--coco_path', default='/data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='/data/detr-workdir/r50-dc5',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', default=False, action='store_true',
                        help='whether to resume from last checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    ## NEW ARG to limit number of evals
    parser.add_argument('--eval_checkpoint_period', default=5, type=int,
                        help="Evaluation and checkpoint period by epoch, defalut is per 10 epochs")

    parser.add_argument('--eval_model_period', default=5, type=int,
                        help="Evaluation period by epoch")

    parser.add_argument('--num_classes', default=21, type=int,
                        help="corresponds to `max_obj_id + 1`, where max_obj_id is the maximum id for a class in your dataset.")
    parser.add_argument('--device_num', default=0, type=int, help='device number')

    parser.add_argument('--wandb_run_name', default='detr_testing', type=str, help='wandb run name')
    parser.add_argument('--wandb_project', default='anchor_detr_revisions', type=str, help='wandb project name')

    parser.add_argument('--only_neck', action='store_true', help='whether to train only neck')
    parser.add_argument('--freeze_backbone', action='store_true', help='whether to train only backbone')
    parser.add_argument('--sam_vit', default='vit_b', type=str, help='which sam_vit to use')

    #paths
    parser.add_argument('--train_path', default='train', type=str)
    parser.add_argument('--val_path', default='val', type=str)
    parser.add_argument('--test_path', default='test', type=str)

    # json root
    parser.add_argument('--json_root', default='annotations', type=str)
    parser.add_argument('--train_json', default='train.json', type=str)
    parser.add_argument('--val_json', default='val.json', type=str)
    parser.add_argument('--test_json', default='test.json', type=str)

    # for ablation purposes
    parser.add_argument('--delete_nuclear_channel', action='store_true', help='whether to delete nuclear channel')
    parser.add_argument('--additional_augmentations', default="none", type=str, choices=['none','more_scales', 'newest_aug_2024', 'newest' ,'random_rotation_and_flip', 'qilin', 'combined', 'combined_v2'], help='whether to delete nuclear channel')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    if isinstance(args.dilation, str):
        if args.dilation.lower() == 'false':
            args.dilation = False
        elif args.dilation.lower() == 'true':
            args.dilation = True
        else:
            raise ValueError("The dilation should be True or False")
    print(args)

    device = torch.device(args.device)
    # device = torch.device(f'{args.device}:{args.device_num}')
    # device = torch.device('cpu')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_name = args.wandb_run_name

    wandb.init(project=args.wandb_project, entity='allcell', name=run_name)

    # define backbone
    # from models.backbone import build_backbone
    # backbone = build_backbone(args)
    #
    # from anchor_testing import build_samanchor, TestAnchorDETR
    # sam_model, criterion, postprocessors = build_samanchor(args)
    # model = TestAnchorDETR(backbone, sam_model)
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set=args.eval_set, args=args)

    ## tmp code for debuging on mac
    # dataset_train = torch.utils.data.Subset(dataset_train, range(4))
    # dataset_val = torch.utils.data.Subset(dataset_val, range(4))

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # data_loader_val = data_loader_train

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
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n,
                                                                                                   args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    # print()
    # print('param_dicts: ')
    # print("len(param_dicts[0]['params']): ", len(param_dicts[0]['params']))
    # print("len(param_dicts[1]['params']): ", len(param_dicts[1]['params']))
    # print("len(param_dicts[2]['params']): ", len(param_dicts[2]['params']))
    # print()

    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    if args.auto_resume:
        if not args.resume:
            args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
        if not os.path.isfile(args.resume):
            args.resume = ''

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']

            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print(
                    'Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if not args.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, save_json=True)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            if utils.is_main_process():
                with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
                    json.dump(coco_evaluator.results['bbox'], f)
        return

    print("Start training")
    # print('output_dir: ', args.output_dir)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.eval_checkpoint_period == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:06}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

                # # Get sample validation data
                # sample_in, sample_tgts = next(iter(data_loader_val))
                #
                # # Draw boxes and original image
                # # data_dir = './data/coco/val/'
                # image_id = int(sample_tgts[0]['image_id'])
                # orig_image_path = dataset_val.coco.loadImgs(image_id)[0]
                # # orig_image = Image.open(os.path.join(str(data_loader_val.dataset.root), orig_image_path['file_name'])) # err for tiff
                # np_img = imread(os.path.join(str(data_loader_val.dataset.root), orig_image_path['file_name'])).astype(
                #     np.float32)
                # uint8_img = (np_img / (np_img.max() + 1e-5) * 255.0).astype('uint8')
                # orig_image = Image.fromarray(uint8_img, 'RGB')
                # draw = ImageDraw.Draw(orig_image, "RGBA")
                #
                # # get annotations and labels
                # annotations = dataset_val.coco.imgToAnns[image_id]
                # cats = dataset_val.coco.cats
                # id2label = {k: v['name'] for k, v in cats.items()}
                #
                # for annotation in annotations:
                #     box = annotation['bbox']
                #     class_idx = int(annotation['category_id'])
                #     x, y, w, h = tuple(box)
                #     draw.rectangle((x, y, x + w, y + h), outline='red', width=1)
                #     # draw.text((x, y), id2label[class_idx], fill='white')
                #
                # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                #
                # ax[0].imshow(orig_image)
                # ax[0].set_title('original image')
                # ax[0].set_xticks([])
                # ax[0].set_yticks([])

                # # Get model outputs
                # with torch.no_grad():
                #     model.to(device)
                #     model.eval()
                #     sample_out = model(sample_in.to(device))
                #     orig_target_sizes = torch.stack([t["size"] for t in sample_tgts], dim=0)
                #     results = postprocessors['bbox'](sample_out, orig_target_sizes.to(device))
                #
                # ## Draw boxes and processed image
                # proc_img, _ = sample_in.decompose()
                # rescaled_img = (proc_img[0] - proc_img[0].min()) / (proc_img[0].max() - proc_img[0].min())
                # image = T.ToPILImage()(rescaled_img)
                # draw = ImageDraw.Draw(image, "RGBA")
                #
                # max_score = 0
                # for i in range(len(results[0]['boxes'])):
                #     box = results[0]['boxes'][i]
                #     score = results[0]['scores'][i]
                #     class_idx = int(results[0]['labels'][i]) - 1
                #
                #     max_score = max(max_score, score.max())
                #
                #     if score > 0.4:
                #         x_min, y_min, x_max, y_max = tuple(box)
                #         draw.rectangle((x_min, y_min, x_max, y_max), outline='red', width=1)
                #         # draw.text((x_min, y_min), id2label[class_idx], fill='white')
                #
                # ax[1].imshow(image)
                # ax[1].set_title('Predictions image')
                # ax[1].set_xticks([])
                # ax[1].set_yticks([])
                # fig.tight_layout()
                # # save figure
                # # fig.savefig(os.path.join(args.output_dir, f'sample_val_fig_epoch_{epoch}.png'))
                #
                # wandb.log({'sample_val_fig': wandb.Image(fig)})
                #
                # plt.close(fig)
                #
                # print()
                # print(f'max box scsore for this round is {max_score:04}')
                # print()

        # if (epoch + 1) % args.eval_model_period == 0:  # per args.eval_checkpoint_period epoch log
        #     test_stats, coco_evaluator = evaluate(
        #         model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        #     )
        #
        #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                  **{f'test_{k}': v for k, v in test_stats.items()},
        #                  'epoch': epoch,
        #                  'n_parameters': n_parameters}
        #
        #     # print(args.output_dir)
        #     if args.output_dir and utils.is_main_process():
        #         with (output_dir / "log.txt").open("a") as f:
        #             f.write(json.dumps(log_stats) + "\n")
        #
        #         if (coco_evaluator is not None):
        #             (output_dir / 'eval').mkdir(exist_ok=True)
        #             if "bbox" in coco_evaluator.coco_eval:
        #                 filenames = ['latest.pth']
        #                 if epoch % args.eval_checkpoint_period == 0:
        #                     filenames.append(f'{epoch:06}.pth')
        #                 for name in filenames:
        #                     torch.save(coco_evaluator.coco_eval["bbox"].eval,
        #                                output_dir / "eval" / name)
        else:  # per epoch log
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            # print(args.output_dir)
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AnchorDETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
