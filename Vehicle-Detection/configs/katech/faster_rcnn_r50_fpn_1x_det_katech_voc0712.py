_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', 
    '../_base_/datasets/katech_voc.py',
    '../_base_/default_runtime.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[9])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=12)  # actual epoch = 4 * 3 = 12

load_from = "https://download.openmmlab.com/mmdetection/v2.0/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth"
