from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from mmdet3d.core.bbox import CameraInstance3DBoxes
from mmcv import Config
import os
import torch

cfg = Config.fromfile("/media/drive-2t/hoangnv83/MV2DFusion_custom/projects/configs/nusc/custom_config.py")
# import modules from plguin/xx, registry will be updated
if hasattr(cfg, 'plugin'):
    if cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            if isinstance(plugin_dir, str):
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            elif isinstance(plugin_dir, (tuple, list)):
                for _plugin_dir in plugin_dir:
                    _module_dir = os.path.dirname(_plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)
        else:
            # import dir is the dirpath for the config file
            _module_dir = os.path.dirname(args.config)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)

dataset = build_dataset(cfg.data.train)

for data in dataset:
    print("KEYS:", data.keys())
    # # # print("IMAGE:", data['img_metas'].data)
    # print(data["gt_bboxes_3d"].data)
    # gt_bboxes_3d = data["gt_bboxes_3d"].data[0].tensor
    
    # print("GT_BBOXES", data["gt_bboxes"].data)
    # for gt_boxs in data["gt_bboxes"].data[0]:
    #     print(len(gt_boxs))
    #     for gt in gt_boxs:
    #         print(gt)
        # break

    # boxes = CameraInstance3DBoxes(data["bboxes3d_cams"][0], box_dim=7)
    # corners = boxes.corners
    
    # N, num_corners, _ = corners.shape
    
    # bbox_corners3d_hom = torch.cat([
    #     corners, torch.ones((N, num_corners, 1), device=corners.device)
    # ], dim=-1)
    
    # print(corners.shape)
    # instrinsics = data["intrinsics"].data
    # print(instrinsics.shape)
    
    # projected = bbox_corners3d_hom @ instrinsics[0][0].T
    # # Chuẩn hóa để lấy (x, y)
    # x = projected[..., 0] / projected[..., 2]
    # y = projected[..., 1] / projected[..., 2]

    # # Stack lại thành (15, 8, 2)
    # bbox_2d_corners = torch.stack([x, y], dim=-1)

    # # Tính min/max để lấy bbox 2D (x1, y1, x2, y2)
    # x1y1 = bbox_2d_corners.min(dim=1).values  # (15, 2)
    # x2y2 = bbox_2d_corners.max(dim=1).values  # (15, 2)
    # bbox2d = torch.cat([x1y1, x2y2], dim=-1)  # (15, 4)  [x1, y1, x2, y2]
    # print(bbox2d)
    
    
    
    # print(data['lidar2img'].data)
    # print(len(data["centers2d"].data[0]))
    # print(len(data["depths"].data[0]))
    
    # print("extrinsics", data["extrinsics"].data.shape)
    # print("intrinsics", data["intrinsics"].data.shape)
    
    # print(data["pair"])
    break
    
    

# model = build_model(
#         cfg.model,
#         train_cfg=cfg.get('train_cfg'),
#         test_cfg=cfg.get('test_cfg'))
# for name, param in model.named_parameters():
    
#     print(f"Param {name}: {param.requires_grad}.")
