from mmdet3d.datasets.pipelines import LoadAnnotations3D
from mmdet3d.datasets import PIPELINES  # Dùng đúng với mmdet3d 1.0.0rc4

@PIPELINES.register_module()
class LoadAnnotations3DWithTokens(LoadAnnotations3D):
    def __init__(self, with_bbox_3d=True, with_label_3d=True, **kwargs):
        super().__init__(with_bbox_3d=with_bbox_3d, with_label_3d=with_label_3d, **kwargs)

    def __call__(self, res):
        # Lấy instance_tokens từ get_data_info (nếu có)
        tokens = res.pop('instance_tokens', None)

        # Gọi hàm gốc
        res = super().__call__(res)

        # Nếu có token và label thì đồng bộ token theo label
        if tokens is not None and 'gt_labels_3d' in res:
            labels = res['gt_labels_3d']
            # Trích tensor nếu là DataContainer
            if hasattr(labels, '_data'):
                labels = labels._data

            # Tạo mask và lọc
            mask = labels != -1
            kept_tokens = [tokens[i] for i, m in enumerate(mask) if m]
            res['instance_tokens'] = kept_tokens

        return res
