import argparse
import logging
import os
import os.path as osp
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from torchvision import transforms
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPVisionModelWithProjection
import imageio
from einops import rearrange

# models
from models.model import GASModel
from models.smpl_encoder import PoseNet
from models.nerf_encoder import NeRFNet
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

# diffusion pipeline
from pipelines.pipeline_gas import GASPipeline as StableVideoDiffusionPipeline

# utils
from utils.video_utils import save_videos_grid



def setup_savedir(cfg):
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if cfg.exp_name is None:
        savedir = f"results/exp-{time_str}"
    else:
        savedir = f"results/{cfg.exp_name}-{time_str}"

    os.makedirs(savedir, exist_ok=True)

    return savedir


def get_mask(mask_path):
    msk = imageio.imread(mask_path)
    msk[msk != 0] = 255
    return msk

def apply_mask(img_pil, mask_path):
    img_array = np.array(img_pil)
    
    # Get the mask and normalize it
    mask = np.array(get_mask(mask_path)) / 255.
    
    # Ensure the mask has the same number of channels as the image
    if mask.ndim == 2:
        mask = np.stack([mask]*3, axis=-1)
    
    # Apply the mask
    img_array[mask == 0] = 0
    
    # Convert back to a PIL Image
    return Image.fromarray(img_array.astype(np.uint8))

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[0]
    latents = vae.encode(t).latent_dist.mode()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    return latents


def inference(
    cfg,
    vae,
    image_enc,
    model,
    smpl_vidpil_lst, # SMPL normal map
    nerf_vidpil_lst, # NeRF rendering
    obs_img, # reference image
    video_length,
    width,
    height,
    device="cuda",
    dtype=torch.float16,
):

    denoising_unet = model.denoising_unet
    pose_net = model.pose_net.to(dtype)
    nerf_net = model.nerf_net.to(dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        cfg.svd_base_path,
        unet=denoising_unet,
        image_encoder=image_enc,
        vae=vae,
        torch_dtype=dtype,
    )
    pipeline = pipeline.to(device, dtype)
    with torch.no_grad():
        pose_latents = pose_net(smpl_vidpil_lst.to(device, dtype))
        nerf_vidpil_lst = pipeline.image_processor.preprocess(nerf_vidpil_lst, height=height, width=width).to(device, dtype)
        nerf_vidpil_lst = tensor_to_vae_latent(nerf_vidpil_lst, vae.to(device, dtype))

        nerf_latents = nerf_net(nerf_vidpil_lst.to(device, dtype))

    video_frames = pipeline(
        image=obs_img,
        pose_latents_all=pose_latents.to(dtype),
        nerf_latents_all = nerf_latents.to(dtype),
        num_frames=video_length,
        num_inference_steps=cfg.num_inference_steps,
        motion_bucket_id=127,
        fps=7,
        noise_aug_strength=0,
        tile_size=video_length, 
        tile_overlap=cfg.frames_overlap,
        generator=generator, 
        min_guidance_scale=1, 
        max_guidance_scale=2.0,
        output_type="pt",
        task='mv'
        ).frames.cpu()

    for vid_idx in range(video_frames.shape[0]):
        # deprecated first frame because of ref image
        _video_frames = video_frames[vid_idx, :]

    del pipeline
    torch.cuda.empty_cache()

    return _video_frames
    

def main(cfg, img_path):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    save_dir = setup_savedir(cfg)
    logging.info(f"Running inference ...")

    # setup pretrained models
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32


    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_model_path).to( 
        "cuda", dtype=weight_dtype
    )

    denoising_unet = UNetSpatioTemporalConditionModel.from_pretrained(
        cfg.unet_path,
        subfolder="unet",
        low_cpu_mem_usage=False,
        device_map=None
    ).to(device="cuda")
    
    pose_net = PoseNet(noise_latent_channels=denoising_unet.config.block_out_channels[0]).to(device="cuda")
    nerf_net = NeRFNet(noise_latent_channels=denoising_unet.config.block_out_channels[0]).to(device="cuda")
    ckpt_dir = cfg.ckpt_dir
    ckpt_files = sorted(os.listdir(ckpt_dir), key=lambda x: int(x.split("-")[-1].split(".")[0]))
    latest_pth_name = (Path(ckpt_dir) / ckpt_files[-1]).stem
    ckpt_step = int(latest_pth_name.split("-")[-1])

    denoising_unet.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, f'unet-{ckpt_step}.pth'),
            map_location="cpu",
        ),
    )
    pose_net.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, f'pose_net-{ckpt_step}.pth'),
            map_location="cpu",
        ),
    )

    nerf_net.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, f'nerf_net-{ckpt_step}.pth'),
            map_location="cpu",
        ),
    )
    model = GASModel(
        denoising_unet,
        pose_net,
        nerf_net,
    ).cuda() 

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    to_tensor = transforms.ToTensor()
    img_name = os.path.basename(img_path).split(".")[0]
    normal_folder = "reference_images_freeview_smpl/"
    nerf_folder = "reference_images_freeview_nerf/"
    all_subjects = [img_name]

    for subject in all_subjects:

        nerf_dir = os.path.join(nerf_folder, subject)
        normal_dir = os.path.join(normal_folder, subject)

        obs_img_path = img_path
        obs_img_mask_path = os.path.join(os.path.dirname(img_path), "groundsam_vis", f"{img_name}.png.mask.png")

        obs_img_pil = Image.open(obs_img_path)
        obs_img_pil = apply_mask(obs_img_pil, obs_img_mask_path)

        smpl_vidpil_lst = []
        nerf_vidpil_lst = []

        camera_trajectory = list(range(20))
        for _, cam_name in enumerate(camera_trajectory):
            # SMPL normal condition
            cam_name = str(cam_name).zfill(2)
            smpl_image_path = os.path.join(normal_dir, 'normal', str(cam_name).zfill(2), f'{img_name}.png')
            smpl_img_pil = Image.open(smpl_image_path)
            smpl_vidpil_lst.append(to_tensor(smpl_img_pil))

            # NeRF rendering condition
            nerf_img_path = os.path.join(nerf_dir, f'pose_0000_view_{str(cam_name).zfill(4)}.png')
            nerf_img = Image.open(nerf_img_path)
            nerf_vidpil_lst.append(to_tensor(nerf_img))


        video_length = cfg.data.video_length
        smpl_vid = torch.stack(smpl_vidpil_lst, dim=0)
        nerf_vid = torch.stack(nerf_vidpil_lst, dim=0)

        result_video_tensor = inference(
            cfg=cfg,
            vae=vae,
            image_enc=image_enc,
            model=model,
            smpl_vidpil_lst=smpl_vid,
            nerf_vidpil_lst=nerf_vid,
            obs_img=obs_img_pil, 
            video_length=video_length,
            width=cfg.width,
            height=cfg.height,
            device="cuda",
            dtype=weight_dtype,
        )

        result_video_tensor = result_video_tensor[None, ...]
        result_video_tensor = rearrange(result_video_tensor, 'b f c h w -> b c f h w')

        obs_video_tensor = to_tensor(obs_img_pil)[None, :, None, ...].repeat(
            1, 1, video_length, 1, 1
        )


        grid_video = torch.cat([obs_video_tensor, result_video_tensor], dim=0)
        save_videos_grid(grid_video, osp.join(save_dir, f"subject_{subject}_grid.mp4"), fps=12)
        
    logging.info(f"Inference completed, results saved in {save_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/inference.yaml")
    parser.add_argument("--img_path", type=str, default="./")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        cfg = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")

    main(cfg, args.img_path)



