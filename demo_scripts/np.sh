#!/bin/bash

PROJECT_ROOT="$PWD"
VIDEO_PATH="assets/demo_videos/demo2.mp4"
VIDEO_DIR=$(dirname "$VIDEO_PATH")
mkdir -p "$VIDEO_DIR/images"
ffmpeg -i "$VIDEO_PATH" -r 12 -c:v png "$VIDEO_DIR/images/%04d.png"

IMAGE_PATH="assets/demo_videos/images/0001.png"

# pad and resize
python utils/resize.py "$VIDEO_DIR/images"



# estimate foreground human mask
python modules/grounded-sam/segment_human.py \
--img_path "$IMAGE_PATH" \
--grounded_checkpoint pretrained_models/Grounded_SAM/groundingdino_swint_ogc.pth \
--sam_checkpoint pretrained_models/Grounded_SAM/sam_vit_h_4b8939.pth


# estimate SMPL parameters
python scripts/data_processors/smpl/estimate_smpl.py --driving_video_path "$VIDEO_DIR"



# render SMPL normals
blender scripts/data_processors/smpl/blend/smpl_rendering.blend --background --python scripts/data_processors/smpl/render_smpl_normal_map_np.py -- --driving_path "$VIDEO_DIR"


# render generalizable human nerf
cd modules/sherf
python -u train.py --outdir=logs/tiktok/ --cfg=TikTok --data="$PROJECT_ROOT/$(dirname "$VIDEO_PATH")" \
--gpus=1 --batch=1 --gamma=5 --aug=noaug --neural_rendering_resolution_initial=512 --gen_pose_cond=True --gpc_reg_prob=0.8 --kimg 800 --workers 0 \
--use_1d_feature True --use_2d_feature True --use_3d_feature True --use_sr_module False --sample_obs_view True --fix_obs_view False --use_nerf_decoder True --use_trans True \
--test_flag True \
--resume $PROJECT_ROOT/pretrained_models/sherf/network-snapshot.pkl \
--test_mode pose

# clear logs
rm -r logs



cd ../..
# run diffusion model
CUDA_VISIBLE_DEVICES=0 python inference_novel_pose.py --config configs/inference/novel_pose.yaml