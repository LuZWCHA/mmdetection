
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
# python tools/test.py   \
# projects/CO_DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_gaiic_dual_stream.py \
# work_dirs/co_dino_5scale_r50_lsj_8xb2_1x_gaiic_dual_stream/epoch_12.pth \
# --resume work_dirs/co_dino_5scale_swin_l_16xb1_1x_coco_dual_more_data/

python tools/test.py   \
projects/CO_DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_gaiic_dual_stream_dual_backbone.py \
work_dirs/co_dino_5scale_r50_lsj_8xb2_1x_gaiic_dual_stream_dual_backbone/epoch_12.pth\