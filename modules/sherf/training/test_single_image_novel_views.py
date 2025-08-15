# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

import os
import numpy as np
import torch
import dnnlib
import imageio
import cv2
from skimage.metrics import structural_similarity

from torch.utils.data import DataLoader
import lpips


img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

loss_fn_vgg = lpips.LPIPS(net='vgg')

#----------------------------------------------------------------------------

def to_cuda(device, sp_input):
    for key in sp_input.keys():
        if torch.is_tensor(sp_input[key]):
            sp_input[key] = sp_input[key].to(device)
        if key=='params':
            for key1 in sp_input['params']:
                if torch.is_tensor(sp_input['params'][key1]):
                    sp_input['params'][key1] = sp_input['params'][key1].to(device)
    
        if key=='t_params':
            for key1 in sp_input['t_params']:
                if torch.is_tensor(sp_input['t_params'][key1]):
                    sp_input['t_params'][key1] = sp_input['t_params'][key1].to(device)

        if key=='obs_params':
            for key1 in sp_input['obs_params']:
                if torch.is_tensor(sp_input['obs_params'][key1]):
                    sp_input['obs_params'][key1] = sp_input['obs_params'][key1].to(device)

    return sp_input

#----------------------------------------------------------------------------

def ssim_metric(rgb_pred, rgb_gt, mask_at_box, H, W):
    # convert the pixels into an image
    img_pred = np.zeros((H, W, 3))
    img_pred[mask_at_box] = rgb_pred
    img_gt = np.zeros((H, W, 3))
    img_gt[mask_at_box] = rgb_gt

    # crop the object region
    x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
    img_pred = img_pred[y:y + h, x:x + w]
    img_gt = img_gt[y:y + h, x:x + w]
    
    # compute the ssim
    ssim = structural_similarity(img_pred, img_gt, multichannel=True)
    lpips = loss_fn_vgg(torch.from_numpy(img_pred).permute(2, 0, 1).to(torch.float32), torch.from_numpy(img_gt).permute(2, 0, 1).to(torch.float32)).reshape(-1).item()

    return ssim, lpips

#----------------------------------------------------------------------------

def test(model, savedir=None, neural_rendering_resolution=128, rank=0, use_sr_module=False, white_back=False, sample_obs_view=False, fix_obs_view=False, dataset_name='RenderPeople', data_root=None, obs_view_lst = [0, 16, 31], nv_pose_start=0, np_pose_start=2, pose_interval=0, pose_num=5,
         tgt_pose_index=0, human_idx_start=0, human_idx_end=500):

    device = torch.device('cuda', rank)
    batch_size = 1
    pose_start = nv_pose_start
    pose_interval = pose_interval
    pose_num = pose_num
    obs_view_lst = obs_view_lst 



    if dataset_name == 'TikTok':
        class_name = 'training.nv_dataset.TikTokDatasetBatch'
        image_scaling = 1

    for obs_view in obs_view_lst:

        human_data_path = data_root

        data_root = human_data_path.strip()
        human_name = os.path.basename(data_root)
        savedir_human = os.path.join(savedir, human_name.split(".")[0]) 

        os.makedirs(savedir_human, exist_ok=True)
        
        test_dataset_kwargs = dnnlib.EasyDict(class_name=class_name, data_root=data_root, split='test', multi_person=False
        , num_instance=1, poses_start=pose_start, poses_interval=pose_interval, poses_num=pose_num, image_scaling=image_scaling, white_back=white_back, sample_obs_view=sample_obs_view, fix_obs_view=fix_obs_view, 
        tgt_pose_index=tgt_pose_index)
        test_set = dnnlib.util.construct_class_by_name(**test_dataset_kwargs) 
        test_set.obs_view_index = obs_view

        pose = 0
        test_set.pose_index = pose
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

        for k, test_data in enumerate(test_loader):
            view_id = k % test_set.num_freeviews
            print("subject: ", human_name, " target view : ", view_id, "pose", pose)

            test_data = to_cuda(device, test_data)
            try:
                gen_img = model(test_data, torch.randn(1, 512).to(device), torch.zeros((1, 25)).to(device), \
                neural_rendering_resolution=neural_rendering_resolution, use_sr_module=use_sr_module, test_flag=True)
            except Exception as e:
                continue

            for j in range(batch_size):

                img_pred = (gen_img['image'][j] / 2 + 0.5).permute(1,2,0)
                pred_filename = os.path.join(savedir_human, 'pose_{:04d}_view_{:04d}.png'.format(int(test_data['pose_index'][j]), view_id))

                imageio.imwrite(pred_filename, to8b(img_pred.cpu().numpy()))

    return