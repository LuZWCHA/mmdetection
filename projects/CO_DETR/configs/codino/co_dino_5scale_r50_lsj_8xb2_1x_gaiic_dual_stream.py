# _base_ = 'mmdet::common/ssj_scp_270k_coco-instance.py'
_base_ = 'mmdet::common/ssj_270k_coco-instance.py'


## Custom ##
# from .my_loading import LoadImageFromFile2
# from .my_wrapper import Image2Broadcaster, Branch
# from .my_formatting import DoublePackDetInputs
custom_imports = dict(imports=['projects.CO_DETR.codetr.codetr_dual_stream',
                               'mmdet.datasets.transforms.my_loading',
                               'mmdet.datasets.transforms.my_wrapper',
                               'mmdet.datasets.transforms.my_formatting',
                               'mmdet.models.data_preprocessors.my_data_preprocessor',
                               'mmdet.datasets.my_coco',
                               'projects.CO_DETR.codetr'
                               ], allow_failed_imports=False)

dataset_type = 'DualStreamCocoDataset'
data_root = '/nasdata/private/zwlu/detection/Gaiic1/projects/data/mmdet/gaiic/GAIIC2024/'
data_root = '/root/workspace/data/GAIIC2024/'
data_root_vis = '/root/workspace/data/DroneVehicle/coco_format/'
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
load_from = 'co_dino_5scale_r50_lsj_8xb2_1x_coco-69a72d67.pth'  # noqa

image_size = (1024, 1024)
num_classes = 5
classes = ('car', 'truck', 'bus', 'van', 'freight_car')

batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size)
]

# model settings
num_dec_layer = 6
loss_lambda = 2.0

model = dict(
    type='CoDETR_Dual',
    # If using the lsj augmentation,
    # it is recommended to set it to True.
    use_lsj=True,
    # detr: 52.1
    # one-stage: 49.4
    # two-stage: 47.9
    eval_module='detr',  # in ['detr', 'one-stage', 'two-stage']
    data_preprocessor=dict(
        type='DoubleInputDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        batch_augments=batch_augments),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5),
    query_head=dict(
        type='CoDINOHead',
        num_query=900,
        num_classes=num_classes,
        in_channels=2048,
        as_two_stage=True,
        dn_cfg=dict(
            label_noise_scale=0.5,
            box_noise_scale=1.0,
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
        transformer=dict(
            type='CoDinoTransformer',
            with_coord_feat=False,
            num_co_heads=2,  # ATSS Aux Head + Faster RCNN Aux Head
            num_feature_levels=5,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                # number of layers that use checkpoint.
                # The maximum value for the setting is num_layers.
                # FairScale must be installed for it to work.
                with_cp=4,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=5,
                        dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=5,
                            dropout=0.0),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(  # Different from the DINO
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0 * num_dec_layer * loss_lambda),
        loss_bbox=dict(
            type='L1Loss', loss_weight=1.0 * num_dec_layer * loss_lambda)),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0 * num_dec_layer * loss_lambda),
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.0 * num_dec_layer * loss_lambda)))
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda),
            loss_bbox=dict(
                type='GIoULoss',
                loss_weight=2.0 * num_dec_layer * loss_lambda),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0 * num_dec_layer * loss_lambda)),
    ],
    # model training and testing settings
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='FocalLossCost', weight=2.0),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ])),
        dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=4000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
    ],
    test_cfg=[
        # Deferent from the DINO, we use the NMS.
        dict(
            max_per_img=300,
            # NMS can improve the mAP by 0.2.
            nms=dict(type='soft_nms', iou_threshold=0.8)),
        dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)),
        dict(
            # atss bbox head:
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100),
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ])

# LSJ + CopyPaste
load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadImageFromFile2'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),

    dict(type='Image2Broadcaster',
        transforms=[
                dict(
                    type='RandomResize',
                    scale=image_size,
                    ratio_range=(0.1, 2.0),
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=image_size,
                    recompute_bbox=True,
                    allow_negative_crop=True),
                dict(type='RandomFlip', prob=0.5),
        ]
    ),
    dict(type='Branch',
         transforms=[
                 dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
        ]
    ),
    
]

train_pipeline = load_pipeline + [
    # dict(type='CopyPaste', max_num_pasted=100),
    
    dict(type='DoublePackDetInputs')
]

# train_dataloader = dict(
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#             pipeline=train_pipeline,
#             type=dataset_type,
#             metainfo=dict(classes=classes),
#             data_root=data_root,
#             ann_file='train.json',
#             data_prefix=dict(img='train/rgb'),
#         )
# )

train_dataloader = dict(
        batch_size=4, num_workers=4, 
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
            type=dataset_type,
            metainfo=dict(classes=classes),
            data_root=data_root,
            ann_file='train.json',
            data_prefix=dict(img='train/rgb'),
            pipeline=train_pipeline
        )
    )

# follow ViTDet
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadImageFromFile2'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),

    dict(type='Branch',
         transforms=[
             dict(type='Resize', scale=image_size, keep_ratio=True),
             dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
         ]),
    
    dict(
        type='DoublePackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'img_path2', 'ori_shape2', 'img_shape2',
                   'scale_factor'))
]


val_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    ann_file=data_root + 'val.json')
# val_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     ann_file=data_root_vis + 'annotations/val_tir.json')
val_dataloader = dict(dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val.json',
        
        data_prefix=dict(img='val/rgb'),
        pipeline=test_pipeline))
# val_dataloader = dict(dataset=dict(
#         type=dataset_type,
#         metainfo=dict(classes=classes),
#         data_root=data_root_vis,
#         ann_file='annotations/val_tir.json',
#         data_prefix=dict(img='images/val/rgb'),
#         pipeline=test_pipeline))
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

test_dataloader = val_dataloader
test_evaluator = val_evaluator

test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=data_root + 'instances_test2017.json',
    outfile_prefix='./dual_test_result'
    )

test_dataloader = dict(dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='instances_test2017.json',
        data_prefix=dict(img='test/rgb'),
        pipeline=test_pipeline))


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone1': dict(lr_mult=0.1), 'backbone2': dict(lr_mult=0.1)}))


max_epochs = 12
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=3))
log_processor = dict(by_epoch=True)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16, enabled = True)

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(
                   type='nms',
                   iou_threshold=0.5),
                   max_per_img=100))

tta_pipeline = [
    dict(type='LoadImageFromFile',
        backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(type='Resize', scale=(1333, 800), keep_ratio=True)
        ], [ # It uses 2 flipping transformations (flipping and not flipping).
            dict(type='RandomFlip', prob=1.),
            dict(type='RandomFlip', prob=0.)
        ], [
            dict(
               type='PackDetInputs',
               meta_keys=('img_id', 'img_path', 'ori_shape',
                       'img_shape', 'scale_factor', 'flip',
                       'flip_direction'))
       ]])]
