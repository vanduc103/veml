"""Faster RCNN with ResNet50-FPN, 1x schedule."""

_base_ = [
    "../_base_/models/faster_rcnn_r50_fpn.py",
    "../_base_/datasets/katech_cocofmt.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

model = dict(roi_head = dict(bbox_head = dict(num_classes=1)))

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
