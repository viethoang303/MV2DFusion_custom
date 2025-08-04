# Copyright (c) Wang, Z
# ------------------------------------------------------------------------
# Modified from StreamPETR (https://github.com/exiawsh/StreamPETR)
# Copyright (c) Shihao Wang
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from mmcv.cnn import Linear, bias_init_with_prob, ConvModule
from mmcv.cnn.bricks.transformer import build_transformer_layer

from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox, normalize_bbox
from mmcv.ops.box_iou_rotated import box_iou_rotated
from mmdet3d.core import nms_bev
from mmdet3d.core.bbox.structures import xywhr2xyxyr

from mmdet.models.utils import NormedLinear
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, \
    nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, \
    SELayer_Linear

from mmdet3d.ops import RoIAwarePool3d, RoIPointPool3d
from mmdet.models import HEADS, LOSSES, build_loss, build_head
from mmcv.runner import ModuleList
from mmdet.core import bbox2roi
from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor

def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb

def boxes3d_to_corners(bboxes3d):
    """Tính 8 góc của bbox3D (N,7) -> (N,8,3)."""
    bboxes3d = bboxes3d[:, :7]  # lấy x, y, z, l, w, h, yaw
    centers = bboxes3d[:, :3]
    dims = bboxes3d[:, 3:6]
    yaws = bboxes3d[:, 6]

    # 8 góc tương đối
    corners_rel = torch.tensor([
        [ 0.5,  0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [ 0.5,  0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [-0.5, -0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ], device=bboxes3d.device).unsqueeze(0)  # [1,8,3]

    corners_rel = corners_rel * dims.unsqueeze(1)  # [N,8,3]

    # rotation quanh trục z
    cosa, sina = torch.cos(yaws), torch.sin(yaws)
    zeros, ones = torch.zeros_like(cosa), torch.ones_like(cosa)
    rot = torch.stack([
        torch.stack([cosa, -sina, zeros], -1),
        torch.stack([sina,  cosa, zeros], -1),
        torch.stack([zeros, zeros, ones], -1)
    ], 1)  # [N,3,3]

    corners = torch.bmm(corners_rel, rot.permute(0,2,1)) + centers.unsqueeze(1)  # [N,8,3]
    return corners

def project_bbox3d_to_2d(bboxes3d, lidar2img, img_shape):
    """
    Project bbox3D từ LiDAR sang từng view camera.
    bboxes3d: (B,N,7 or 10)
    lidar2img: (B,V,4,4)
    img_shape: (H,W)
    return: (B,V,N,4) x1,y1,x2,y2
    """
    B, N, D = bboxes3d.shape
    _, V, _, _ = lidar2img.shape
    H, W = img_shape
    boxes_2d = []

    for b in range(B):
        corners = boxes3d_to_corners(bboxes3d[b])  # [N,8,3]
        corners_hom = torch.cat([corners, torch.ones((N,8,1), device=bboxes3d.device)], -1)  # [N,8,4]

        view_boxes = []
        for v in range(V):
            proj = torch.matmul(lidar2img[b, v], corners_hom.permute(0,2,1))  # [N,4,8]
            proj = proj.permute(0,2,1)
            u = proj[:,:,0] / proj[:,:,2]
            v_ = proj[:,:,1] / proj[:,:,2]

            x1 = torch.clamp(u.min(1).values, 0, W)
            y1 = torch.clamp(v_.min(1).values, 0, H)
            x2 = torch.clamp(u.max(1).values, 0, W)
            y2 = torch.clamp(v_.max(1).values, 0, H)
            view_boxes.append(torch.stack([x1,y1,x2,y2], -1))  # [N,4]
        view_boxes = torch.stack(view_boxes,0)  # [V,N,4]
        boxes_2d.append(view_boxes)
    return torch.stack(boxes_2d,0)  # [B,V,N,4]


@LOSSES.register_module()
class Contrastive3D2DLoss(nn.Module):
    def __init__(self, dim2d=256, dim3d=128, proj_dim=128, temperature=0.1, loss_weight=0.1):
        super().__init__()
        self.proj2d = nn.Sequential(
            nn.Linear(dim2d, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, proj_dim)
        )
        # Projection head cho 3D
        self.proj3d = nn.Sequential(
            nn.Linear(dim3d, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, proj_dim)
        )
        self.temperature = temperature
        self.loss_weight = loss_weight

    def forward(self, feat2d, feat3d):
        """
        feat2d: [N, C2, 7, 7]
        feat3d: [N, 7, 7, 7, C3]
        """
        N = feat2d.shape[0]

        # 1. Pooling để mỗi box còn 1 vector
        feat2d = F.adaptive_avg_pool2d(feat2d, 1).view(N, -1)    # [N, C2]
        feat3d = F.adaptive_avg_pool3d(feat3d.permute(0,4,1,2,3), 1).view(N, -1)  # [N, C3]

        # 2. Projection
        z2d = self.proj2d(feat2d.float())   # [N, proj_dim]
        z3d = self.proj3d(feat3d.float())   # [N, proj_dim]

        # 3. Normalize
        z2d = F.normalize(z2d, dim=-1)
        z3d = F.normalize(z3d, dim=-1)

        # 4. InfoNCE logits
        logits = z2d @ z3d.T / self.temperature  # [N, N]
        labels = torch.arange(N, device=feat2d.device)
        loss = F.cross_entropy(logits, labels) * self.loss_weight
        return loss

@LOSSES.register_module()
class SupCon3D2DLoss(nn.Module):
    def __init__(self, dim2d=256, dim3d=128, proj_dim=128, temperature=0.1, loss_weight=0.1):
        super().__init__()
        self.proj2d = nn.Sequential(
            nn.Linear(dim2d, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )
        self.proj3d = nn.Sequential(
            nn.Linear(dim3d, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )
        self.temperature = temperature
        self.loss_weight = loss_weight

    def forward(self, feat2d, feat3d, group_ids):
        """
        feat2d: [M, C2, 7, 7]   -> RoI features (multi-view)
        feat3d: [N, 7, 7, 7, C3] -> Voxel features (1 per object)
        group_ids: [M] tensor, chỉ định RoI này thuộc object nào (0..N-1)
        """
        M = feat2d.shape[0]
        N = feat3d.shape[0]

        # 1. Pooling
        feat2d = F.adaptive_avg_pool2d(feat2d, 1).view(M, -1)  # [M, C2]
        feat3d = F.adaptive_avg_pool3d(feat3d.permute(0,4,1,2,3), 1).view(N, -1)  # [N, C3]

        # 2. Projection
        z2d = F.normalize(self.proj2d(feat2d.float()), dim=-1)  # [M, proj_dim]
        z3d = F.normalize(self.proj3d(feat3d.float()), dim=-1)  # [N, proj_dim]

        # 3. Tính similarity giữa tất cả voxel và tất cả RoI
        logits = torch.matmul(z3d, z2d.T) / self.temperature  # [N, M]

        # 4. Tạo mask multi-positive: voxel i positive với tất cả RoI có group_ids == i
        mask = torch.zeros((N, M), device=feat2d.device)
        for i in range(N):
            mask[i, group_ids == i] = 1

        # 5. SupCon loss
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()  # ổn định số học
        exp_logits = torch.exp(logits) * (1 - torch.eye(N, M, device=feat2d.device)[:,:M])

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = - mean_log_prob_pos.mean()

        return loss * self.loss_weight

# class SupCon3D2DLoss(nn.Module):
#     def __init__(self, dim2d=256, dim3d=128, proj_dim=128, temperature=0.1, loss_weight=0.1):
#         super().__init__()
#         self.proj2d = nn.Sequential(
#             nn.Linear(dim2d, proj_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(proj_dim, proj_dim)
#         )
#         self.proj3d = nn.Sequential(
#             nn.Linear(dim3d, proj_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(proj_dim, proj_dim)
#         )
#         self.temperature = temperature
#         self.loss_weight = loss_weight

#     def forward(self, feat2d, feat3d, group_ids):
#         """
#         feat2d: [M, C2, 7, 7]   -> RoI features
#         feat3d: [N, 7, 7, 7, C3] -> Voxel features (flatten theo batch)
#         group_ids: [M] tensor, chỉ định RoI thuộc object nào (theo index đã flatten)
#         """
#         M = feat2d.shape[0]
#         N = feat3d.shape[0]

#         # Pooling
#         feat2d = F.adaptive_avg_pool2d(feat2d, 1).view(M, -1)          # [M, C2]
#         feat3d = F.adaptive_avg_pool3d(feat3d.permute(0,4,1,2,3), 1).view(N, -1)  # [N, C3]

#         # Projection
#         z2d = F.normalize(self.proj2d(feat2d.float()), dim=-1)  # [M, proj_dim]
#         z3d = F.normalize(self.proj3d(feat3d.float()), dim=-1)  # [N, proj_dim]

#         # Similarity
#         logits = torch.matmul(z3d, z2d.T) / self.temperature  # [N, M]

#         # Mask: multi-positive
#         mask = torch.zeros((N, M), device=feat2d.device)
#         mask[group_ids, torch.arange(M, device=feat2d.device)] = 1

#         # Contrastive loss
#         logits_max, _ = torch.max(logits, dim=1, keepdim=True)
#         logits = logits - logits_max.detach()
#         exp_logits = torch.exp(logits)
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

#         mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
#         loss = - mean_log_prob_pos.mean()
#         return loss * self.loss_weight

'''
MAIN LOSS FUNCTION FOR SUPCON 3D-2D
'''
# @LOSSES.register_module()
# class SupCon3D2DLoss(nn.Module):
#     def __init__(self, dim2d=256, dim3d=128, proj_dim=128, temperature=0.1, loss_weight=0.1):
#         super().__init__()
#         self.proj2d = nn.Sequential(
#             nn.Linear(dim2d, proj_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(proj_dim, proj_dim)
#         )
#         self.proj3d = nn.Sequential(
#             nn.Linear(dim3d, proj_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(proj_dim, proj_dim)
#         )
#         self.temperature = temperature
#         self.loss_weight = loss_weight

#     def forward(self, feat2d, feat3d, group_ids):
#         """
#         feat2d: [M, C2, 7, 7]
#         feat3d: either
#             - [N, D, H, W, C3]  (raw voxel, need pool), or
#             - [N, C3]          (pooled_aligned passed in)
#         group_ids: [M] flat indices
#         """
#         M = feat2d.size(0)

#         # 1) 2D pooling
#         feat2d = F.adaptive_avg_pool2d(feat2d, 1).view(M, -1)  # [M, C2]

#         # 2) 3D pooling if needed
#         if feat3d.ndim == 5:
#             # original code path
#             N = feat3d.size(0)
#             # permute to [N, C3, D, H, W]
#             f3 = feat3d.permute(0,4,1,2,3)
#             feat3d_vec = F.adaptive_avg_pool3d(f3, 1).view(N, -1)  # [N, C3]
#         elif feat3d.ndim == 2:
#             # already pooled: shape [N, C3]
#             feat3d_vec = feat3d
#             N = feat3d_vec.size(0)
#         else:
#             raise RuntimeError(f"feat3d must be 2D or 5D but got {feat3d.shape}")

#         # 3) Validate group_ids
#         assert group_ids.min() >= 0 and group_ids.max() < N, (
#             f"group_ids must be in [0, {N-1}], got {group_ids.min()}–{group_ids.max()}"
#         )

#         # 4) Projection + normalize
#         z2d = F.normalize(self.proj2d(feat2d.float()), dim=-1)       # [M, D]
#         z3d = F.normalize(self.proj3d(feat3d_vec.float()), dim=-1)   # [N, D]

#         # 5) Compute similarity [N, M]
#         logits = z3d @ z2d.T / self.temperature

#         # 6) Build positive mask
#         mask = torch.zeros((N, M), device=logits.device)
#         mask[group_ids, torch.arange(M, device=logits.device)] = 1

#         # 7) SupCon loss
#         logits = logits - logits.max(1, keepdim=True).values
#         exp_logits = torch.exp(logits)
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
#         mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
#         loss = - mean_log_prob_pos.mean()

#         return loss * self.loss_weight


# @LOSSES.register_module()
# class SupCon3D2DLoss(nn.Module):
#     def __init__(self, dim2d=256, dim3d=128, proj_dim=128, temperature=0.1, loss_weight=0.1):
#         super().__init__()
#         self.proj2d = nn.Sequential(
#             nn.Linear(dim2d, proj_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(proj_dim, proj_dim)
#         )
#         self.proj3d = nn.Sequential(
#             nn.Linear(dim3d, proj_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(proj_dim, proj_dim)
#         )
#         self.temperature = temperature
#         self.loss_weight = loss_weight

#     def forward(self, feat2d: torch.Tensor, feat3d: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
#         # Global average pooling for 2D features
#         if feat2d.ndim == 4:
#             feat2d = F.adaptive_avg_pool2d(feat2d, 1).squeeze(-1).squeeze(-1)
#         if feat3d.ndim == 5:
#             feat3d = feat3d.permute(0,4,1,2,3)
#             feat3d = F.adaptive_avg_pool3d(feat3d, 1).squeeze(-1).squeeze(-1).squeeze(-1)

#         # Align 3D features by group ids
#         feat3d_aligned = feat3d[group_ids]  # hoặc feat3d[group_ids] nếu cần matching cụ thể
#         print(feat3d_aligned.shape)
#         # --- Loại bỏ ROI không hợp lệ ---
#         valid_mask = ~torch.isnan(feat2d).any(dim=1) & (torch.norm(feat2d, dim=1) > 1e-6)
#         feat2d = feat2d[valid_mask]
#         feat3d_aligned = feat3d_aligned[valid_mask]
#         group_ids = group_ids[valid_mask]

#         # Nếu sau khi lọc không còn ROI hợp lệ thì trả về 0
#         if feat2d.size(0) == 0:
#             return torch.tensor(0., device=feat2d.device)

#         # Projection and normalization
#         z2d = F.normalize(self.proj2d(feat2d.float()), dim=1, eps=1e-6)
#         z3d = F.normalize(self.proj3d(feat3d_aligned.float()), dim=1, eps=1e-6)

#         # Contrastive logits between 2D and 3D features
#         logits = torch.matmul(z3d, z2d.t()) / self.temperature

#         # Log-softmax (ổn định số học)
#         log_prob = F.log_softmax(logits, dim=1)

#         # Mask positives (diagonal: same index pairs)
#         mask = torch.eye(logits.size(0), device=logits.device)

#         mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
#         loss = -mean_log_prob_pos.mean()
#         return loss * self.loss_weight



@HEADS.register_module()
class MV2DFusionHead(AnchorFreeHead):
    _version = 2
    TRACKING_CLASSES = ['car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 embed_dims=256,
                 num_query=100,
                 num_reg_fcs=2,
                 memory_len=6 * 256,
                 topk_proposals=256,
                 num_propagated=256,
                 with_dn=True,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None,
                 transformer=None,
                 normedlinear=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                #  loss_contrastive=dict(type='SupConLoss', loss_weight=1.0),
                loss_contrastive=None,
                # contrastive_head = None, 
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner3D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0)), ),
                 test_cfg=dict(max_per_img=100),
                 # denoise config
                 scalar=5,
                 noise_scale=0.4,
                 noise_trans=0.0,
                 dn_weight=1.0,
                 split=0.5,
                 # image query config
                 prob_bin=50,
                 # nms config
                 post_bev_nms_thr=0.2,
                 post_bev_nms_score=0.0,
                 post_bev_nms_ops=[],
                 # init config
                 init_cfg=None,
                 debug=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.code_weights = self.code_weights[:self.code_size]

        if match_costs is not None:
            self.match_costs = match_costs
        else:
            self.match_costs = self.code_weights

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is MV2DFusionHead):
            assert isinstance(class_weight, float), 'Expected ' \
                                                    'class_weight to have type float. Found ' \
                                                    f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                                                     'bg_cls_weight to have type float. Found ' \
                                                     f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided ' \
                                            'when train_cfg is set.'
            assigner = train_cfg['assigner']

            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.match_with_velo = match_with_velo
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.embed_dims = embed_dims

        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split

        self.act_cfg = transformer.get('act_cfg', dict(type='ReLU', inplace=True))
        self.num_pred = transformer['decoder']['num_layers']
        self.normedlinear = normedlinear
        self.prob_bin = prob_bin

        super(MV2DFusionHead, self).__init__(num_classes, in_channels, init_cfg=init_cfg)
        self.roi = RoIAwarePool3d(
            out_size=7,
            max_pts_per_voxel=128,
            mode='max'
        )
        self.roi2d = SingleRoIExtractor(
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=-1),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64]
        )

        self.contrastive_loss = build_loss(loss_contrastive)# ModuleList([build_loss(loss_contrastive) for _ in range(6)])
        
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.transformer = build_transformer(transformer)

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights), requires_grad=False)

        self.match_costs = nn.Parameter(torch.tensor(
            self.match_costs), requires_grad=False)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.pc_range = nn.Parameter(torch.tensor(
            self.bbox_coder.pc_range), requires_grad=False)

        # nms config
        self.post_bev_nms_thr = post_bev_nms_thr
        self.post_bev_nms_score = post_bev_nms_score
        self.post_bev_nms_ops = post_bev_nms_ops

        self._init_layers()
        self.reset_memory()

        self.fp16_enabled = False

        self.debug = debug

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        self.reference_points = nn.Embedding(self.num_query, 3)
        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)

        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.spatial_alignment = MLN(14, use_ln=False)

        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

        # encoding ego pose
        if self.with_ego_pos:
            self.ego_pose_pe = MLN(180)
            self.ego_pose_memory = MLN(180)

        # image distribution query positional encoding
        prob_bin = self.prob_bin
        self.dyn_q_embed = nn.Embedding(1, self.embed_dims)
        self.dyn_q_enc = MLN(256)
        self.dyn_q_pos = nn.Sequential(
            nn.Linear(prob_bin * 3, self.embed_dims * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 4, self.embed_dims),
        )
        self.dyn_q_pos_with_prob = SELayer_Linear(self.embed_dims, in_channels=prob_bin)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, prob_bin))
        reg_branch = nn.Sequential(*reg_branch)
        self.dyn_q_prob_branch = nn.ModuleList([
            copy.deepcopy(reg_branch) for _ in range(self.num_pred)
        ])

        # point cloud embedding
        self.pts_embed = nn.Sequential(
            nn.Linear(128, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.pts_query_embed = nn.Sequential(
            nn.Linear(128, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.pts_q_embed = nn.Embedding(1, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.num_propagated > 0:
            nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False

        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None

    def pre_update_memory(self, data):
        x = data['prev_exists']
        B = x.size(0)
        # refresh the memory when the scene changes
        if self.memory_embedding is None or data['timestamp'].size(0) != self.memory_embedding.size(0):
            self.memory_embedding = x.new_zeros(B, self.memory_len, self.embed_dims)
            self.memory_reference_point = x.new_zeros(B, self.memory_len, 3)
            self.memory_timestamp = x.new_zeros(B, self.memory_len, 1)
            self.memory_egopose = x.new_zeros(B, self.memory_len, 4, 4)
            self.memory_velo = x.new_zeros(B, self.memory_len, 2)
            self.memory_query_mask = x.new_zeros(B, self.memory_len, 1, dtype=torch.bool)
            self.memory_instance_inds = x.new_zeros(B, self.memory_len) - 1
        else:
            self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
            self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose_inv'],
                                                                     reverse=False)
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.memory_velo = memory_refresh(self.memory_velo[:, :self.memory_len], x)
            self.memory_query_mask = memory_refresh(self.memory_query_mask[:, :self.memory_len], x.bool())
            self.memory_instance_inds = memory_refresh(self.memory_instance_inds[:, :self.memory_len], x, value=-1)

        # for the first frame, padding pseudo_reference_points (non-learnable)
        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points.weight * (
                        self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            self.memory_reference_point[:, :self.num_propagated] = \
                self.memory_reference_point[:, :self.num_propagated] + (1 - x).view(B, 1, 1) * pseudo_reference_points
            self.memory_egopose[:, :self.num_propagated] = self.memory_egopose[:, :self.num_propagated] + \
                                                           (1 - x).view(B, 1, 1, 1) * torch.eye(4, device=x.device)
            self.memory_query_mask[:, :self.num_propagated] = \
                self.memory_query_mask[:, :self.num_propagated] | (1 - x).view(B, 1, 1).bool()

    def post_update_memory(self, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict,
                           query_mask=None, instance_inds=None, ):
        if self.training and mask_dict and mask_dict['pad_size'] > 0:
            rec_reference_points = all_bbox_preds[:, :, mask_dict['pad_size']:, :3][-1]
            rec_velo = all_bbox_preds[:, :, mask_dict['pad_size']:, -2:][-1]
            rec_memory = outs_dec[:, :, mask_dict['pad_size']:, :][-1]
            rec_score = all_cls_scores[:, :, mask_dict['pad_size']:, :][-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        else:
            rec_reference_points = all_bbox_preds[..., :3][-1]
            rec_velo = all_bbox_preds[..., -2:][-1]
            rec_memory = outs_dec[-1]
            rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)

        rec_score[~query_mask] = 0

        # topk proposals
        topk_score, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(rec_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()

        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_egopose = torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
        self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose'],
                                                                 reverse=False)
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose
        if query_mask is not None:
            query_mask = torch.gather(query_mask[..., None], 1, topk_indexes)
            self.memory_query_mask = torch.cat([query_mask, self.memory_query_mask], dim=1)
        if instance_inds is not None:
            instance_inds = torch.gather(instance_inds, 1, topk_indexes[..., 0])
            self.memory_instance_inds = torch.cat([instance_inds, self.memory_instance_inds], dim=1)

    @staticmethod
    def transform3d(pose, coords3d):
        coords3d = torch.cat([coords3d, torch.ones_like(coords3d[..., 0:1])], dim=-1)   # B, ..., 4
        shape = coords3d.shape[:-1]
        new_shape = [shape[i] if i == 0 else 1 for i in range(len(shape))]
        pose = pose.view(*new_shape, 4, 4)
        transformed_coords3d = (pose @ coords3d[..., None])[..., :3, 0]
        return transformed_coords3d

    @staticmethod
    def rotate2d(pose, coords2d):
        shape = coords2d.shape[:-1]
        new_shape = [shape[i] if i == 0 else 1 for i in range(len(shape))]
        pose = pose.view(*new_shape, 4, 4)[..., :2, :2]
        rotated_coords2d = (pose @ coords2d[..., None])[..., 0]
        return rotated_coords2d

    @staticmethod
    def get_box_info(bbox_preds):
        bbox_x, bbox_y, bbox_w, bbox_l, bbox_o = bbox_preds[..., 0], bbox_preds[..., 1], bbox_preds[..., 3], \
                                                 bbox_preds[..., 4], bbox_preds[..., 6]
        bbox_z, bbox_h = bbox_preds[..., 2], bbox_preds[..., 5]
        # bbox_o = -(bbox_o + np.pi / 2)
        bbox_o = (bbox_o + np.pi / 2)
        center = torch.stack([bbox_x, bbox_y], dim=-1)
        cos, sin = torch.cos(bbox_o), torch.sin(bbox_o)
        pc0 = torch.stack([bbox_x + cos * bbox_l / 2 + sin * bbox_w / 2,
                           bbox_y + sin * bbox_l / 2 - cos * bbox_w / 2], dim=-1)
        pc1 = torch.stack([bbox_x + cos * bbox_l / 2 - sin * bbox_w / 2,
                           bbox_y + sin * bbox_l / 2 + cos * bbox_w / 2], dim=-1)
        pc2 = 2 * center - pc0
        pc3 = 2 * center - pc1

        xyxyo = torch.stack([pc0, pc1, pc2, pc3, center], dim=-2)   # [..., 5, 2]
        bbox_z = bbox_z[..., None, None].expand_as(xyxyo[..., :1])
        xyxyo = torch.cat([xyxyo, bbox_z], dim=-1)
        return xyxyo, torch.stack([bbox_w, bbox_l, bbox_h], dim=-1), torch.stack([cos, sin], dim=-1)

    def temporal_alignment(self, query_pos, tgt, reference_points):
        B = query_pos.size(0)

        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (
                    self.pc_range[3:6] - self.pc_range[0:3])
        temp_pos = self.query_embedding(pos2posemb3d(temp_reference_point))
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)

        if self.with_ego_pos:
            rec_ego_motion = torch.cat(
                [torch.zeros_like(reference_points[..., :3]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)
            memory_ego_motion = torch.cat(
                [self.memory_velo, self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[..., :1])))
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())

        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat([reference_points, temp_reference_point[:, :self.num_propagated]], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[
                1] + self.num_propagated, 1, 1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]

        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training and self.with_dn:
            targets = [
                torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),
                          dim=1) for img_meta in img_metas]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            # gt_num
            known_num = [t.size(0) for t in targets]

            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0),), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            # groups = min(self.scalar, self.num_query // max(known_num))
            known_indice = known_indice.repeat(self.scalar, 1).view(-1)
            known_labels = labels.repeat(self.scalar, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(self.scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.scalar, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                               diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:3] = (known_bbox_center[..., 0:3] - self.pc_range[0:3]) / (
                            self.pc_range[3:6] - self.pc_range[0:3])

                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = self.num_classes

            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.scalar)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            if reference_points.dim() == 2:
                padded_reference_points = \
                    torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
            elif reference_points.dim() == 3:
                padded_reference_points = torch.cat([padding_bbox.unsqueeze(0).repeat(batch_size, 1, 1), reference_points], dim=1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.scalar)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(
                    reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == self.scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            # update dn mask for temporal modeling
            query_size = pad_size + self.num_query + self.num_propagated
            tgt_size = pad_size + self.num_query + self.memory_len
            temporal_attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
            temporal_attn_mask[:attn_mask.size(0), :attn_mask.size(1)] = attn_mask
            temporal_attn_mask[pad_size:, :pad_size] = True
            attn_mask = temporal_attn_mask

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }
        else:
            if reference_points.dim() == 2:
                padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            elif reference_points.dim() == 3:
                padded_reference_points = reference_points
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is MV2DFusionHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                      unexpected_keys, error_msgs)

    def gen_dynamic_query(self, static_query, dynamic_query, dynamic_query_feats=None):
        B = len(dynamic_query)
        zero = static_query.sum() * 0
        max_len = max(x.size(0) for x in dynamic_query)
        max_len = max(max_len, 1)
        query_coords = static_query.new_zeros((B, max_len, dynamic_query[0].size(1), 3))
        query_probs = static_query.new_zeros((B, max_len, dynamic_query[0].size(1)))
        query_ref = static_query.new_zeros((B, max_len, 3)) + zero + 0.5
        query_mask = static_query.new_zeros((B, max_len), dtype=torch.bool)
        query_feats = static_query.new_zeros((B, max_len, self.embed_dims))
        self.num_query = max_len

        for b in range(B):
            dyn_q = dynamic_query[b][..., :3].clone()
            dyn_q[..., 0:3] = (dyn_q[..., 0:3] - self.pc_range[0:3]) / (
                    self.pc_range[3:6] - self.pc_range[0:3])
            dyn_q_prob = dynamic_query[b][..., 3]
            ref_point = (dyn_q_prob[:, None] @ dyn_q)[:, 0]
            query_coords[b, :dyn_q.size(0)] = dyn_q
            query_probs[b, :dyn_q.size(0)] = dyn_q_prob
            query_ref[b, :dyn_q.size(0)] = ref_point
            query_mask[b, :dyn_q.size(0)] = 1
            if dynamic_query_feats is not None:
                query_feats[b, :dyn_q.size(0)] = dynamic_query_feats[b][:dyn_q.size(0)]

        return query_ref, query_coords, query_probs, query_feats, query_mask

    def gen_pts_query(self, pts_query_center):
        pts_ref = pts_query_center.clone()
        pts_ref[..., 0:3] = (pts_ref[..., 0:3] - self.pc_range[0:3]) / (
                self.pc_range[3:6] - self.pc_range[0:3])
        self.num_query += pts_ref.size(1)
        return pts_ref

    def forward(self, img_metas, dyn_query=None, dyn_feats=None,
                pts_query_center=None, pts_query_feat=None, pts_feat=None, pts_pos=None, **data):
        print("MV@DFusion HEAD", pts_query_feat.shape, pts_feat.shape)
        # zero init the memory bank
        self.pre_update_memory(data)

        # process image feats
        intrinsics = data['intrinsics'] / 1e3
        extrinsics = data['extrinsics'][..., :3, :]
        mln_input = torch.cat([intrinsics[..., 0,0:1], intrinsics[..., 1,1:2], extrinsics.flatten(-2)], dim=-1) # Đại loại concat intrinsic và extrínics với nhau
        mln_input = mln_input.flatten(0, 1).unsqueeze(1)
        mlvl_feats = data['img_feats_for_det'] # mlvl_feats chứa các feats từ các view khác nhau của camera
        B, N, _, _, _ = mlvl_feats[0].shape
        feat_flatten_img = []
        spatial_flatten_img = []
        for i in range(1, len(mlvl_feats)): # for từng từng feats 
            B, N, C, H, W = mlvl_feats[i].shape
            mlvl_feat = mlvl_feats[i].reshape(B * N, C, -1).transpose(1, 2)
            mlvl_feat = self.spatial_alignment(mlvl_feat, mln_input) # Điều chỉnh image feats phù hợp với từng view camera, tức là ngoài thông tin feature thì nó có thêm thông tin về camera được trích xuất ở góc nhìn camera nào
            feat_flatten_img.append(mlvl_feat.to(torch.float))
            spatial_flatten_img.append((H, W))
        feat_flatten_img = torch.cat(feat_flatten_img, dim=1)
        spatial_flatten_img = torch.as_tensor(spatial_flatten_img, dtype=torch.long, device=mlvl_feats[0].device)
        level_start_index_img = torch.cat((spatial_flatten_img.new_zeros((1, )), spatial_flatten_img.prod(1).cumsum(0)[:-1])) # Sử dụng cho deformable attention

        # process point cloud feats
        feat_flatten_pts = self.pts_embed(pts_feat) # embed pc feat, chỉ là vài linear layer
        pos_flatten_pts = pts_pos

        # generate image query
        reference_points, query_coords, query_probs, query_feats, query_mask = \
            self.gen_dynamic_query(self.reference_points.weight, dyn_query, dyn_feats.get('query_feats', None))

        # generate point cloud query
        pts_ref = self.gen_pts_query(pts_query_center)
        query_mask = torch.cat([torch.ones_like(pts_ref[..., 0]).bool(), query_mask], dim=1)
        reference_points = torch.cat([pts_ref, reference_points], dim=1)

        num_query_img = int(self.num_query - pts_ref.size(1))
        num_query_pts = pts_ref.size(1)

        # denoise training
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(B, reference_points, img_metas)

        # mask out padded query for attention
        tgt_size = self.num_query + self.num_propagated
        src_size = self.num_query + self.memory_len
        if attn_mask is None:
            attn_mask = torch.zeros((tgt_size, src_size), dtype=torch.bool, device=reference_points.device)
        pad_size = attn_mask.size(0) - tgt_size
        if mask_dict is not None:
            assert pad_size == mask_dict['pad_size']
        attn_mask = attn_mask.repeat(B, 1, 1)
        tgt_query_mask = torch.cat([query_mask, self.memory_query_mask[:, :self.num_propagated, 0]], dim=1)
        src_query_mask = torch.cat([query_mask, self.memory_query_mask[:, :, 0]], dim=1)
        attn_mask[:, :, pad_size:] = ~src_query_mask[:, None]
        num_heads = self.transformer.decoder.layers[0].attentions[0].num_heads
        attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)

        # query content feature
        tgt = self.dyn_q_embed.weight.repeat(B, num_query_img, 1)
        pts_tgt = self.pts_q_embed.weight.repeat(B, num_query_pts, 1)
        tgt = torch.cat([tgt.new_zeros((B, pad_size, self.embed_dims)), pts_tgt, tgt], dim=1)
        pad_query_feats = query_feats.new_zeros([B, pad_size + self.num_query, self.embed_dims])
        pts_query_feat = self.pts_query_embed(pts_query_feat)
        pad_query_feats[:, pad_size:pad_size + num_query_pts] = pts_query_feat
        pad_query_feats[:, pad_size + num_query_pts:pad_size + self.num_query] = query_feats
        tgt = self.dyn_q_enc(tgt, pad_query_feats)

        # query positional encoding
        query_pos = self.query_embedding(pos2posemb3d(reference_points))
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = \
            self.temporal_alignment(query_pos, tgt, reference_points)

        # encode position distribution for image query
        query_pos_det = self.dyn_q_pos(query_coords.flatten(-2, -1))
        query_pos_det = self.dyn_q_pos_with_prob(query_pos_det, query_probs)
        query_pos[:, pad_size + num_query_pts:pad_size + self.num_query] = query_pos_det

        dyn_q_mask = torch.zeros_like(tgt[..., 0]).bool()
        dyn_q_mask[:, pad_size + num_query_pts:pad_size + self.num_query] = 1
        dyn_q_mask[:, pad_size:] &= tgt_query_mask
        dyn_q_mask_img = dyn_q_mask[:, pad_size + num_query_pts:pad_size + self.num_query]
        dyn_q_coords = query_coords[dyn_q_mask_img]
        dyn_q_probs = query_probs[dyn_q_mask_img]

        # transformer decoder
        outs_dec, reference_points, dyn_q_logits = self.transformer(
            tgt, query_pos, attn_mask,
            feat_flatten_img, spatial_flatten_img, level_start_index_img, self.pc_range, img_metas, data['lidar2img'],
            feat_flatten_pts=feat_flatten_pts, pos_flatten_pts=pos_flatten_pts,
            temp_memory=temp_memory, temp_pos=temp_pos,
            cross_attn_masks=None, reference_points=reference_points,
            dyn_q_coords=dyn_q_coords, dyn_q_probs=dyn_q_probs, dyn_q_mask=dyn_q_mask, dyn_q_pos_branch=self.dyn_q_pos,
            dyn_q_pos_with_prob_branch=self.dyn_q_pos_with_prob, dyn_q_prob_branch=self.dyn_q_prob_branch,
        )

        # generate prediction
        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points[lvl].clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (
                    all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])

        # mask out padded query for output
        all_cls_scores[:, :, pad_size:][:, ~tgt_query_mask] = -40 + all_cls_scores[:, :, pad_size:][:, ~tgt_query_mask] * 0
        all_bbox_preds[:, :, pad_size:][:, ~tgt_query_mask] = 0 + all_bbox_preds[:, :, pad_size:][:, ~tgt_query_mask] * 0

        # apply nms for post-processing
        iou_thr = self.post_bev_nms_thr
        score_thr = self.post_bev_nms_score
        ops = self.post_bev_nms_ops
        if len(ops) > 0:
            bbox_output = denormalize_bbox(all_bbox_preds[-1, :, pad_size:], None)
            bbox_bev = bbox_output[..., [0, 1, 3, 4, 6]]
            score_bev = all_cls_scores[-1, :, pad_size:].sigmoid().max(-1).values.clone()
            score_bev[~tgt_query_mask] = 0
            nms_tgt_query_mask = torch.zeros_like(tgt_query_mask)
            for i in range(B):
                if 0 in ops:
                    assert len(ops) == 1
                    # 0. all -> all nms
                    all_boxes = bbox_bev[i]
                    all_scores = score_bev[i]
                    keep = nms_bev(xywhr2xyxyr(all_boxes), all_scores, iou_thr, pre_max_size=None, post_max_size=None)
                    nms_tgt_query_mask[i, :][keep] = 1
                    nms_tgt_query_mask[i, :][all_scores == 0] = 0

            nms_tgt_query_mask &= all_cls_scores[-1, :, pad_size:].sigmoid().max(-1).values > score_thr
            tgt_query_mask &= nms_tgt_query_mask

            all_cls_scores[:, :, pad_size:][:, ~tgt_query_mask] = -40
            all_bbox_preds[:, :, pad_size:][:, ~tgt_query_mask] = 0

        # update the memory bank
        self.post_update_memory(data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict,
                                query_mask=tgt_query_mask, instance_inds=None, )

        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            all_cls_scores = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            all_bbox_preds = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
            mask_dict['output_known_lbs_bboxes'] = (output_known_class, output_known_coord)
        else:
            mask_dict = None

        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'dyn_cls_scores': all_cls_scores,
            'dyn_bbox_preds': all_bbox_preds,
            'dn_mask_dict': mask_dict,
        }
        return outs

    def prepare_for_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long().cpu()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indexes for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indexes for each image.
                - neg_inds (Tensor): Sampled negative indexes for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore, self.match_costs, self.match_with_velo)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        # DETR
        if sampling_result.num_gts > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg, pos_inds_list)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds',))
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    pts_feats=None,
                    pts_xyz=None,
                    img_feats=None,
                    lidar2img=None,
                    image_shape=None,
                    gt_bboxes_ignore_list=None, index=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # num_imgs = cls_scores.size(0)
        num_imgs = len(cls_scores)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        # used for debug
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, pos_inds_list) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        # print("BBOX WEIGHTs:", bbox_weights.shape, bbox_targets.shape, labels.shape, pos_inds_list)

        if pts_feats is not None:
            print("LOSS SINGLE", pts_feats.shape, pos_inds_list)
            bbox_preds_ = bbox_preds[:,pos_inds_list[0],:]
            for bbox in bbox_preds_:
                print(bbox.shape)
            # print("CHECK pre", bbox_preds_.shape, lidar2img.shape, bbox_preds_.max())
            # try:
            bbox_projected = project_bbox3d_to_2d(bbox_preds_, lidar2img, image_shape)
            print("bbox_projected", bbox_projected.shape, bbox_projected.max())
            # Check bbox sanity
            if torch.isnan(bbox_projected).any():
                print("NaN found in bbox_projected!")
            
            # Extract features
            for fet in img_feats:
                print("IN LOSS SINGLE:", fet.shape, fet.max(), fet.min())
            """o day tiep"""
            
            """
            # bbox_projected: [B, V, N, 4]
            B, V, N, D = bbox_projected.shape
            rois_by_view = []

            # Bước 1: Chuẩn bị RoIs cho từng view
            valid_indices_all = []
            for v in range(V):
                view_boxes = []
                valid_indices_view = []
                for b in range(B):
                    boxes = bbox_projected[b, v]  # [N, 4]
                    valid_mask = ~torch.all(boxes == 0, dim=-1)
                    boxes = boxes[valid_mask]
                    indices = torch.nonzero(valid_mask).squeeze(-1)
                    view_boxes.append(boxes)
                    valid_indices_view.append(indices)
                rois = bbox2roi(view_boxes)
                rois_by_view.append(rois)
                valid_indices_all.append(valid_indices_view)

            # Bước 2: Trích xuất ROI features
            roi_feats_by_view = []
            for v in range(V):
                # Giả sử img_feats là list [level1, level2, ...], mỗi level: [B, V, C, H, W]
                feats_for_view = [lvl_feats[v, :, :, :].unsqueeze(0) for lvl_feats in img_feats]  # List[B, C, H, W]
                
                # Debug
                for i, f in enumerate(feats_for_view):
                    print(f"Level {i} view {v} feature shape:", f.shape)
                print("ROI levels:", rois_by_view[v][:, 0].unique())

                roi_feats = self.roi2d(feats_for_view, rois_by_view[v])  # [num_rois, C, 7, 7]
                print("CHECK ROI each view:", roi_feats.shape, roi_feats.max(), roi_feats.min())
                roi_feats_by_view.append(roi_feats)
            """

            
            pooled_features = self.roi(
                rois=bbox_preds_[0, :, :7].contiguous(),               # (1, M, 7)
                pts=pts_xyz,             # (1, N, 3)
                pts_feature=pts_feats,     # (1, N, C)
            )
            
            """
            loss_contrastive = 0.0
            for head_idx, head in enumerate(self.contrastive_head):
                head_losses = []
                for v in range(V):   # num_views = 5
                    roi_feats = roi_feats_by_view[v]  # [num_rois, C, 7, 7]
                    # print("HẸ HẸ", roi_feats.shape, pooled_features[valid_indices_all[v]].shape)
                    loss = head(roi_feats, pooled_features[valid_indices_all[v]])
                    head_losses.append(loss)
                # Cộng hoặc trung bình loss của 5 view
                # losses[f'aux_head{head_idx}_loss'] = torch.stack(head_losses).mean()
                loss_contrastive += torch.stack(head_losses).mean()
            """
            # bbox_projected: [B, V, N, 4]
            B, V, N, _ = bbox_projected.shape
            image_h, image_w = image_shape

            rois_by_view = []
            group_ids_all = []

            for v in range(V):
                view_boxes = []
                group_ids_view = []

                for b in range(B):
                    boxes = bbox_projected[b, v]

                    # --- Clip vào frame & loại box rỗng ---
                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, image_w - 1)
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, image_h - 1)
                    w = boxes[:, 2] - boxes[:, 0]
                    h = boxes[:, 3] - boxes[:, 1]
                    valid_mask = (w > 1) & (h > 1) & ~torch.all(boxes == 0, dim=-1)
                    boxes = boxes[valid_mask]
                    indices = torch.nonzero(valid_mask).squeeze(-1)

                    if boxes.numel() == 0:
                        continue

                    view_boxes.append(boxes)
                    group_ids_view.append(indices + b * N)  # shift index theo batch

                if len(view_boxes) == 0:
                    continue
                rois = bbox2roi(view_boxes)
                rois_by_view.append(rois)
                group_ids_all.append(torch.cat(group_ids_view))

            # --- Trích xuất RoI features ---
            all_roi_feats = []
            all_group_ids = []
            for v, rois in enumerate(rois_by_view):
                feats_for_view = [lvl_feats[v, :, :, :].unsqueeze(0) for lvl_feats in img_feats]  # lấy feature từng view
                roi_feats = self.roi2d(feats_for_view, rois)  # [num_rois, C, 7, 7]
                all_roi_feats.append(roi_feats)
                all_group_ids.append(group_ids_all[v])

            if len(all_roi_feats) == 0:
                roi_feats_final = torch.empty(0, device=bbox_projected.device)
                group_ids_final = torch.empty(0, dtype=torch.long, device=bbox_projected.device)
            else:
                roi_feats_final = torch.cat(all_roi_feats, dim=0)
                group_ids_final = torch.cat(all_group_ids, dim=0)

                        # --- Lấy voxel features tương ứng ---
            # pooled_aligned = pooled_features.view(-1, pooled_features.shape[-1])[group_ids_final]  # [M, C3]

            # --- Contrastive Loss ---
            loss_contrastive = self.contrastive_loss(
                roi_feats_final,        # [M, C, 7,7]
                pooled_features,          # [M, C3]
                group_ids_final
            )


            """phan tach o day """
            # # … trước đó bạn đã có:
            # pooled_features = self.roi(
            #     rois=bbox_preds_[0, :, :7].contiguous(),  # (1, M, 7)
            #     pts=pts_xyz,                              # (1, N, 3)
            #     pts_feature=pts_feats,                    # (1, N, C3)
            # )  # pooled_features: [1, N, C3]

            # # --- 1) Hãy detach pooled_features để không ảnh hưởng backward của detection ---
            # pooled_feats = pooled_features#.detach()  # [1, N, C3]

            # # --- 2) Chuẩn bị các RoI 2D độc lập cho contrastive ---
            # B, V, N, _ = bbox_projected.shape
            # img_h, img_w = image_shape
            # all_roi_feats = []
            # all_group_ids = []

            # for v in range(V):
            #     view_boxes = []
            #     group_ids_v = []
            #     for b in range(B):
            #         boxes = bbox_projected[b, v].clone()
            #         # Clip và filter
            #         boxes[:, [0,2]] = boxes[:, [0,2]].clamp(0, img_w-1)
            #         boxes[:, [1,3]] = boxes[:, [1,3]].clamp(0, img_h-1)
            #         w = boxes[:,2] - boxes[:,0]
            #         h = boxes[:,3] - boxes[:,1]
            #         valid = (w>1) & (h>1)
            #         idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            #         if idx.numel()==0:
            #             continue
            #         view_boxes.append(boxes[valid])
            #         # shift index để flatten theo batch
            #         group_ids_v.append(b * N + idx)
            #     if not view_boxes:
            #         continue
            #     rois = bbox2roi(view_boxes)  # [M_v,5]
            #     roi_feats = self.roi2d(
            #         [lvl_feats[v].unsqueeze(0) for lvl_feats in img_feats],
            #         rois
            #     )  # [M_v, C2, 7,7]
            #     all_roi_feats.append(roi_feats)
            #     all_group_ids.append(torch.cat(group_ids_v))

            # if len(all_roi_feats) > 0:
            #     # Gộp tất cả view
            #     roi_feats_final = torch.cat(all_roi_feats, dim=0)      # [M_tot, C2,7,7]
            #     group_ids_final = torch.cat(all_group_ids, dim=0)      # [M_tot]

            #     # --- 3) Lấy đúng pooled voxel features tương ứng ---
            #     # pooled_feats: [1, N, C3] -> flatten thành [B*N, C3]
            #     pooled_flat = pooled_feats.view(-1, pooled_feats.size(-1))  # [B*N, C3]
            #     # index ra [M_tot, C3]
            #     pooled_aligned = pooled_flat[group_ids_final]               # [M_tot, C3]

            #     # --- 4) Tính contrastive loss ---
            #     with torch.cuda.amp.autocast(enabled=False):
            #         loss_contrastive = self.contrastive_loss(
            #             roi_feats_final,     # [M_tot, C2,7,7]
            #             pooled_aligned,      # [M_tot, C3]
            #             group_ids_final % N  # object index trong mỗi sample
            #         )
            # else:
            #     loss_contrastive = pooled_feats.new_tensor(0.)


            '''MAIN'''
            # print("LOSS SINGLE", pts_feats.shape, pos_inds_list)
            # # --- Lấy bbox theo từng batch ---
            # bbox_selected = []
            # for b in range(bbox_preds.size(0)):  # B
            #     inds = pos_inds_list[b]
            #     if inds.numel() > 0:
            #         bbox_selected.append(bbox_preds[b:b+1, inds, :])  # (1, M_b, 7)
            #     else:
            #         bbox_selected.append(bbox_preds.new_zeros((1, 0, bbox_preds.size(-1))))
            # bbox_preds_ = torch.cat(bbox_selected, dim=1)  # [1, M_total, 7] để dùng chung với self.roi

            # for bbox in bbox_preds_:
            #     print(bbox.shape)

            # # --- Project 3D box sang 2D ---
            # bbox_projected = project_bbox3d_to_2d(bbox_preds_, lidar2img, image_shape)

            # # --- Pooled voxel features ---
            # pooled_features = self.roi(
            #     rois=bbox_preds_[0, :, :7].contiguous(),  # (1, M_total, 7)
            #     pts=pts_xyz,                              # (1, N, 3)
            #     pts_feature=pts_feats,                    # (1, N, C3)
            # )  # [1, N, C3]
            # pooled_feats = pooled_features  # Không detach để contrastive cũng backprop

            # # --- Chuẩn bị RoI 2D ---
            # B, V, N, _ = bbox_projected.shape
            # img_h, img_w = image_shape
            # all_roi_feats = []
            # all_group_ids = []

            # for v in range(V):
            #     view_boxes = []
            #     group_ids_v = []
            #     for b in range(B):
            #         boxes = bbox_projected[b, v].clone()
            #         # Clip và filter
            #         boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, img_w - 1)
            #         boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, img_h - 1)
            #         w = boxes[:, 2] - boxes[:, 0]
            #         h = boxes[:, 3] - boxes[:, 1]
            #         valid = (w > 1) & (h > 1)
            #         idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            #         if idx.numel() == 0:
            #             continue
            #         view_boxes.append(boxes[valid])
            #         group_ids_v.append(b * N + idx)  # Flatten index
            #     if not view_boxes:
            #         continue
            #     rois = bbox2roi(view_boxes)  # [M_v, 5]
            #     roi_feats = self.roi2d(
            #         [lvl_feats[v].unsqueeze(0) for lvl_feats in img_feats],
            #         rois
            #     )  # [M_v, C2, 7,7]
            #     all_roi_feats.append(roi_feats)
            #     all_group_ids.append(torch.cat(group_ids_v))

            # if len(all_roi_feats) > 0:
            #     roi_feats_final = torch.cat(all_roi_feats, dim=0)  # [M_tot, C2,7,7]
            #     group_ids_final = torch.cat(all_group_ids, dim=0)  # [M_tot]

            #     # --- Align pooled features ---
            #     pooled_flat = pooled_feats.view(-1, pooled_feats.size(-1))  # [B*N, C3]
            #     pooled_aligned = pooled_flat[group_ids_final]               # [M_tot, C3]

            #     # --- Contrastive loss ---
            #     with torch.cuda.amp.autocast(enabled=False):
            #         loss_contrastive = self.contrastive_loss(
            #             roi_feats_final,     # [M_tot, C2,7,7]
            #             pooled_aligned,      # [M_tot, C3]
            #             group_ids_final % N  # object index trong mỗi sample
            #         )
            # else:
            #     loss_contrastive = pooled_feats.new_tensor(0.)
            '''trên là batch 1'''
            # print(pos_inds_list)
            # B = bbox_preds.size(0)
            # # --- 1) Prepare 3D RoIs ---
            # rois3d_list = []
            # for b in range(B):
            #     inds = pos_inds_list[b]
            #     if len(inds) == 0:
            #         continue
            #     boxes_b = bbox_preds[b, inds]  # [M_b,7]
            #     idx_col = boxes_b.new_full((boxes_b.size(0),1), b)
            #     rois3d_list.append(torch.cat([idx_col, boxes_b[:, :7]], dim=1))
            # if not rois3d_list:
            #     return torch.tensor(0., device=bbox_preds.device)
            # rois3d = torch.cat(rois3d_list, dim=0)  # [M_total,8]

            # # --- 2) Pool 3D features (force FP32 to avoid FP16 overflow) ---
            # with torch.cuda.amp.autocast(enabled=False):
            #     pooled_3d = self.roi(rois3d, pts_xyz.float(), pts_feats.float())  # [M_total, C3]

            # # --- 3) Project and pool 2D features ---
            # V = lidar2img.size(1)
            # h_img, w_img = image_shape
            # all_feats2d = []
            # group_ids = []
            # offset = 0
            # for b in range(B):
            #     inds = pos_inds_list[b]
            #     if len(inds) == 0:
            #         continue
            #     boxes_b = bbox_preds[b, inds]
            #     proj_b = project_bbox3d_to_2d(boxes_b.unsqueeze(0), lidar2img[b:b+1], image_shape)
            #     proj_b = proj_b.squeeze(0)  # [V, M_b,4]
            #     for v in range(V):
            #         xy = proj_b[v]
            #         # clip and filter
            #         xy[:, [0,2]] = xy[:, [0,2]].clamp(0, w_img-1)
            #         xy[:, [1,3]] = xy[:, [1,3]].clamp(0, h_img-1)
            #         w = xy[:,2] - xy[:,0]; h = xy[:,3] - xy[:,1]
            #         valid = (w > 1) & (h > 1)
            #         if not valid.any():
            #             continue
            #         keeps = valid.nonzero(as_tuple=False).squeeze(-1)
            #         batch_idx = xy.new_full((keeps.size(0),1), b)
            #         rois2d = torch.cat([batch_idx, xy[keeps]], dim=1)
            #         # pool 2D (force FP32)
            #         feats_v = [lvl_feats[v:v+1] for lvl_feats in img_feats]
            #         with torch.cuda.amp.autocast(enabled=False):
            #             feat2d = self.roi2d(feats_v, rois2d)
            #         all_feats2d.append(feat2d)
            #         group_ids.append(keeps + offset)
            #     offset += boxes_b.size(0)
            # if not all_feats2d:
            #     return torch.tensor(0., device=bbox_preds.device)
            # feats2d = torch.cat(all_feats2d, dim=0)  # [M2, C2, h, w]
            # group_ids = torch.cat(group_ids)

            # # --- 4) Flatten, normalize and guard NaN/Inf ---
            # # M2, C2, h2, w2 = feats2d.shape
            # # f2d = feats2d.view(M2, C2 * h2 * w2).float()
            # # f3d = pooled_3d.view(-1, pooled_3d.size(-1)).float()
            # # # Normalize
            # # f2d = F.normalize(f2d, dim=1)
            # # f3d = F.normalize(f3d, dim=1)
            # # Replace NaN/Inf
            # # f2d = torch.nan_to_num(f2d, nan=0.0, posinf=1e3, neginf=-1e3)
            # # f3d = torch.nan_to_num(f3d, nan=0.0, posinf=1e3, neginf=-1e3)


            # # Sửa lại để thấy rõ mapping 2D-3D và không index trùng
            # print("HẸ HẸ", feats2d.shape, pooled_3d.shape)
            # # pooled_3d_unique = pooled_3d[group_ids.unique()]
            # # print("HẸ HẸ", feats2d.shape, pooled_3d_unique.shape)
            # with torch.cuda.amp.autocast(enabled=False):
            #     loss_contrastive = self.contrastive_loss(
            #             feats2d,     # [M_tot, C2,7,7]
            #             pooled_3d,      # [M_tot, C3]
            #             group_ids # object index trong mỗi sample
            #         )


        else: 
            loss_contrastive = torch.tensor(0.0, device=gt_labels_list[0].device)


        

        # classification loss
        if isinstance(cls_scores, (tuple, list)):
            cls_scores = torch.cat(cls_scores, dim=0)
        else:
            cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        # print("CHECK POINT 2")
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        if len(cls_scores) == 0:
            loss_cls = cls_scores.sum() * cls_avg_factor
        else:
            loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        if isinstance(bbox_preds, (tuple, list)):
            bbox_preds = torch.cat(bbox_preds, dim=0)
        else:
            bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox, loss_contrastive

    def dn_loss_single(self,
                       cls_scores,
                       bbox_preds,
                       known_bboxs,
                       known_labels,
                       num_total_pos=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 3.14159 / 6 * self.split * self.split * self.split  ### positive rate
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             pts_feats,
             pts_xyz,
             img_feats,
             lidar2img,
             img_shape,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        # print("loss bbox preds:", all_bbox_preds.shape)
        num_dec_layers = len(all_cls_scores)
        print("num_dec_layers", num_dec_layers)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        pts_feats_ = torch.stack([pts_feats for _ in range(num_dec_layers)], dim=0)
        pts_xyz_ = torch.stack([pts_xyz for _ in range(num_dec_layers)], dim=0)
        img_feats_ = [img_feats for _ in range(num_dec_layers)]
        lidar2img_ = torch.stack([lidar2img for _ in range(num_dec_layers)], dim=0)
        image_shape_ = [img_shape for _ in range(num_dec_layers)]

        self.assigner.layer_indicator = 1
        losses_cls, losses_bbox, loss_contrastive = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, 
            pts_feats_,
            pts_xyz_,
            img_feats_,
            lidar2img_,
            image_shape_,
            all_gt_bboxes_ignore_list, list(range(num_dec_layers)))
        self.assigner.layer_indicator = 0

        loss_dict = dict()
        print("CHECK CONTRASTIVE", loss_contrastive)
        loss_dict['loss_contrastive'] = torch.stack(loss_contrastive).sum()
        # loss_dict['size_loss'] = size_loss
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        if preds_dicts['dn_mask_dict'] is not None:
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_loss(
                preds_dicts['dn_mask_dict'])
            all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]
            all_known_labels_list = [known_labels for _ in range(num_dec_layers)]
            all_num_tgts_list = [
                num_tgt for _ in range(num_dec_layers)
            ]

            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.dn_loss_single, output_known_class, output_known_coord,
                all_known_bboxs_list, all_known_labels_list,
                all_num_tgts_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                               dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                num_dec_layer += 1

        elif self.with_dn:
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list,
                pts_feats_, img_feats_, lidar2img_,
                all_gt_bboxes_ignore_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1].detach()
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1].detach()
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                               dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i.detach()
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i.detach()
                num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list

