#!/bin/bash

IMAGE_PATH="/data/yixing/GAS/demo_images/image_(3).png"

# check if this image file exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: File '$IMAGE_PATH' not found."
    exit 1
fi

# pad and resize the input image to 512x512
python utils/resize.py "$IMAGE_PATH"



# estimate foreground human mask
python annotator/grounded-sam/segment_human.py \
--img_path "$IMAGE_PATH" \
--grounded_checkpoint pretrained_models/Grounded_SAM/groundingdino_swint_ogc.pth \
--sam_checkpoint pretrained_models/Grounded_SAM/sam_vit_h_4b8939.pth


# estimate SMPL parameters
python scripts/data_processors/smpl/estimate_smpl.py --reference_img_path "$IMAGE_PATH"



# render SMPL normals
/data/yixing/blender-3.6.0-linux-x64/blender scripts/data_processors/smpl/blend/smpl_rendering.blend --background --python scripts/data_processors/smpl/render_smpl_normal_map_nv.py --driving_path demo_images --reference_path "$IMAGE_PATH"




# render generalizable human nerf
cd modules/sherf
python -u train.py --outdir=logs/tiktok/ --cfg=TikTok --data="$IMAGE_PATH" --gpus=1 --batch=1 --gamma=5 --aug=noaug --neural_rendering_resolution_initial=512 --gen_pose_cond=True --gpc_reg_prob=0.8 --kimg 800 --workers 0 --use_1d_feature True --use_2d_feature True --use_3d_feature True --use_sr_module False --sample_obs_view True --fix_obs_view False --use_nerf_decoder True --use_trans True --test_flag True \
--resume ../../pretrained_models/sherf/network-snapshot-001340.pkl \
--test_mode freeview

# clear logs
rm -r logs



cd ../..
# run diffusion model
CUDA_VISIBLE_DEVICES=0 accelerate launch inference_novel_views.py --config configs/inference/novel_views.yaml --img_path "$IMAGE_PATH"