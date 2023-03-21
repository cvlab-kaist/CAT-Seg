import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import copy

def _get_pascal_voc_meta():
    voc_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    voc_colors = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                  [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
                  [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
                  [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
                  [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]
    ret = {
        "stuff_classes" : voc_classes,
        "stuff_colors" : voc_colors,
    }
    return ret

def register_all_pascal_voc(root):
    root = os.path.join(root, "VOCdevkit/VOC2012")
    meta = _get_pascal_voc_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("test", "JPEGImages", "annotations_detectron2"),
        ("test_background", "JPEGImages", "annotations_detectron2_bg"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname, 'val')
        name = f"voc_2012_{name}_sem_seg"

        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='jpg'))
        if "background" in name:
            MetadataCatalog.get(name).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="sem_seg_background", ignore_label=255,
                                          stuff_classes=meta["stuff_classes"] + ["background"], stuff_colors=meta["stuff_colors"])
        else:
            MetadataCatalog.get(name).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=255, **meta,)

def register_all_pascal_voc_background(root):
    root = os.path.join(root, "VOCdevkit/VOC2012")
    meta = _get_pascal_voc_meta()
    meta["stuff_classes"] = meta["stuff_classes"] + ["background"]
    for name, image_dirname, sem_seg_dirname in [
        ("test_background", "image", "label_openseg_background20"),
    ]:
        image_dir = os.path.join(root, image_dirname, 'validation')
        gt_dir = os.path.join(root, sem_seg_dirname, 'validation')
        name = f"voc_2012_{name}_sem_seg"
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='jpg'))
        MetadataCatalog.get(name).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="sem_seg_background", ignore_label=255, **meta,)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_pascal_voc(_root)
#register_all_pascal_voc_background(_root)