"""Faster RCNN with ResNet50-FPN, 1x schedule."""

_base_ = [
    "../_base_/models/faster_rcnn_r50_fpn.py",
    "../_base_/datasets/katech_bdd.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

# update num classes
model = dict(roi_head = dict(bbox_head = dict(num_classes=1)))

load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_1x_det_bdd100k.pth"
