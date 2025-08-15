import torch.nn as nn
from .unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from .smpl_encoder import PoseNet
from .nerf_encoder import NeRFNet

class GASModel(nn.Module):
    def __init__(
        self,
        denoising_unet: UNetSpatioTemporalConditionModel,
        pose_net: PoseNet,
        nerf_net: NeRFNet,
    ):
        super().__init__()

        self.denoising_unet = denoising_unet
        self.pose_net = pose_net
        self.nerf_net = nerf_net

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        nerf_image_latents,
        class_labels,
        clip_image_embeds,
        added_time_ids,
    ):
        pose_latents = self.pose_net(ref_image_latents)
        nerf_latents = self.nerf_net(nerf_image_latents)
            
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=clip_image_embeds,
            added_time_ids=added_time_ids,
            pose_latents=pose_latents,
            nerf_latents=nerf_latents,
            class_labels=class_labels
        ).sample

        return model_pred
    
