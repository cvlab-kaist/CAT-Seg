# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom

import numpy as np
from einops import rearrange
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

@META_ARCH_REGISTRY.register()
class CATSegDemo(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json

        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "visual" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    params.requires_grad = True if "attn" in name or "position" in name else False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

        finetune_backbone = backbone_multiplier > 0.
        for name, params in self.backbone.named_parameters():
            if "norm0" in name:
                params.requires_grad = False
            else:
                params.requires_grad = finetune_backbone

        self.sliding_window = sliding_window
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)
        self.sequential = False

        self.use_sam = False
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(self.device)

        amg_kwargs = {
            "points_per_side": 32,
            "points_per_batch": None,
            "pred_iou_thresh": 0.0,
            "stability_score_thresh": 0.0,
            "stability_score_offset": None,
            "box_nms_thresh": None,
            "crop_n_layers": None,
            "crop_nms_thresh": None,
            "crop_overlap_ratio": None,
            "crop_n_points_downscale_factor": None,
            "min_mask_region_area": None,
        }
        amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
        self.mask = SamAutomaticMaskGenerator(self.sam, output_mode="binary_mask", **amg_kwargs)
        self.overlap_threshold = 0.8
        self.panoptic_on = False

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        sam_images = images
        if not self.training and self.sliding_window:
            if not self.sequential:
                with _ignore_torch_cuda_oom():
                    return self.inference_sliding_window(batched_inputs)
                self.sequential = True
            return self.inference_sliding_window(batched_inputs)

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
        
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        clip_images = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)

        images_resized = F.interpolate(images.tensor, size=(384, 384), mode='bilinear', align_corners=False,)
        features = self.backbone(images_resized)

        outputs = self.sem_seg_head(clip_features, features)

        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
            
            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value

            outputs = outputs.permute(0,2,3,1)
            _targets = torch.zeros(outputs.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot
            
            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            losses = {"loss_sem_seg" : loss}
            return losses
        else:
            image_size = images.image_sizes[0]
            if self.use_sam:
                masks = self.mask.generate(np.uint8(sam_images[0].permute(1, 2, 0).cpu().numpy()))
                outputs, sam_cls = self.discrete_semantic_inference(outputs, masks, image_size)
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])

            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]
            return processed_results


    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs, kernel=384, overlap=0.333, out_res=[640, 640]):
        
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
        sam_images = [image]
        image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
        image = torch.cat((image, global_image), dim=0)

        images = (image - self.pixel_mean) / self.pixel_std
        clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
        clip_images = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)
        
        if self.sequential:
            outputs = []
            for clip_feat, image in zip(clip_features, images):
                feature = self.backbone(image.unsqueeze(0))
                output = self.sem_seg_head(clip_feat.unsqueeze(0), feature)
                outputs.append(output[0])
            outputs = torch.stack(outputs, dim=0)
        else:
            features = self.backbone(images)
            outputs = self.sem_seg_head(clip_features, features)

        outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
        outputs = outputs.sigmoid()
        
        global_output = outputs[-1:]
        global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
        outputs = outputs[:-1]
        outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        catseg_outputs = sem_seg_postprocess(outputs[0], out_res, height, width)

        masks = self.mask.generate(np.uint8(sam_images[0].permute(1, 2, 0).cpu().numpy()))
        if self.use_sam:
            outputs, sam_cls = self.discrete_semantic_inference(outputs, masks, out_res)

        output = sem_seg_postprocess(outputs[0], out_res, height, width)
        
        ret = [{'sem_seg': output}]

        return ret
    
    def discrete_semantic_inference(self, outputs, masks, image_size):
        catseg_outputs = F.interpolate(outputs, size=image_size, mode="bilinear", align_corners=True) #.argmax(dim=1)[0].cpu()
        sam_outputs = torch.zeros_like(catseg_outputs).cpu()
        catseg_outputs = catseg_outputs.argmax(dim=1)[0].cpu()
        sam_classes = torch.zeros(len(masks))
        for i in range(len(masks)):
            m = masks[i]['segmentation']
            s = masks[i]['stability_score']
            idx = catseg_outputs[m].bincount().argmax()
            sam_outputs[0, idx][m] = s
            sam_classes[i] = idx

        return sam_outputs, sam_classes
    