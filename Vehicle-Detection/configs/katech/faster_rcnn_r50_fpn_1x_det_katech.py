"""Faster RCNN with ResNet50-FPN, 1x schedule."""

_base_ = [
    "../_base_/models/faster_rcnn_r50_fpn.py",
    "../_base_/datasets/katech.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

# update num classes
model = dict(roi_head = dict(bbox_head = dict(num_classes=1)))

