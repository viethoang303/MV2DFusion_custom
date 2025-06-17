import torch
import os
from mmcv.parallel import collate, scatter
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.apis import init_model, inference_detector
from mmcv.parallel import DataContainer

cfg = Config.fromfile("projects/configs/nusc/mv2dfusion-fsd_freeze-convnextl_1600_gridmask-ep24_nusc.py")

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

model = build_model(
        cfg.model)
        # train_cfg=cfg.get('train_cfg'),
        # test_cfg=cfg.get('test_cfg'))



load_checkpoint(model, "mv2dfusion-fsd_freeze-convnextl_1600_gridmask-ep48_trainval_mdropout_nusc.py/latest.pth", map_location="cuda:1")
model.to("cuda:1")
model.eval()
print("Success")

dataset = build_dataset(cfg.data.val)

dataloader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=1,
    shuffle=False,
    dist=False
)

def unwrap_data_container(data):
    """Loại bỏ DataContainer trong dict lồng."""
    def unwrap(x):
        if isinstance(x, DataContainer):
            return x.data[0]  # thường là batch size 1
        elif isinstance(x, dict):
            return {k: unwrap(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [unwrap(i) for i in x]
        else:
            return x
    return {k: unwrap(v) for k, v in data.items()}

for data in dataloader:
    print(data.keys())
    data = scatter(data, [1])[0]
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    break