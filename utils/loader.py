"""
TODO
https://github.com/Tencent/MimicMotion/blob/main/mimicmotion/utils/loader.py#L15
"""
import logging

import torch
import torch.utils.checkpoint
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from models.svd.pose_net import PoseNet
from models.svd.champ_model import ChampModel
from models.svd.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from pipelines.svd.pipeline_svd import StableVideoDiffusionPipeline

logger = logging.getLogger(__name__)
