from torch.utils.data import Dataset
import numpy as np
np.random.seed(1314)
import os
import imageio
import cv2
from smpl.smpl_numpy import SMPL

from scipy.spatial.transform import Rotation

import torchvision.transforms as transforms



def _update_extrinsics(
        extrinsics, 
        angle, 
        trans=None, 
        rotate_axis='y'):
    """ Uptate camera extrinsics when rotating it around a standard axis.

    Args:
        - extrinsics: Array (4, 4)
        - angle: Float
        - trans: Array (3, )
        - rotate_axis: String

    Returns:
        - Array (3, 3)
    """
    E = extrinsics
    inv_E = np.linalg.inv(E)

    camrot = inv_E[:3, :3]
    campos = inv_E[:3, 3]
    if trans is not None:
        campos -= trans

    rot_y_axis = camrot.T[1, 1]
    if rot_y_axis < 0.:
        angle = -angle
    
    rotate_coord = {
        'x': 0, 'y': 1, 'z':2
    }
    grot_vec = np.array([0., 0., 0.])
    grot_vec[rotate_coord[rotate_axis]] = angle
    grot_mtx = cv2.Rodrigues(grot_vec)[0].astype('float32')

    rot_campos = grot_mtx.dot(campos) 
    rot_camrot = grot_mtx.dot(camrot)
    if trans is not None:
        rot_campos += trans
    
    return rot_camrot.T, -rot_camrot.T.dot(rot_campos)


def rotate_camera_by_frame_idx(
    extrinsics, 
    frame_idx, 
    trans=None,
    rotate_axis='y',
    period=196,
    inv_angle=False):
    r""" Get camera extrinsics based on frame index and rotation period.

    Args:
        - extrinsics: Array (4, 4)
        - frame_idx: Integer
        - trans: Array (3, )
        - rotate_axis: String
        - period: Integer
        - inv_angle: Boolean (clockwise/counterclockwise)

    Returns:
        - Array (3, 3)
    """
    rotate_axis = 'y'
    angle = 2 * np.pi * (frame_idx / period)
    if inv_angle:
        angle = -angle
    return _update_extrinsics(
                extrinsics, angle, trans, rotate_axis)
def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    ray_d[ray_d==0.0] = 1e-8
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2

    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box

def sample_ray_RenderPeople_batch(img, msk, K, R, T, bounds, image_scaling, white_back=False):

    H, W = img.shape[:2]
    H, W = int(H * image_scaling), int(W * image_scaling)

    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

    K_scale = np.copy(K)
    K_scale[:2, :3] = K_scale[:2, :3] * image_scaling
    ray_o, ray_d = get_rays(H, W, K_scale, R, T)
    pose = np.concatenate([R, T], axis=1)

    bound_mask = get_bound_2d_mask(bounds, K_scale, pose, H, W)
    mask_bkgd = True

    msk = msk * bound_mask
    if mask_bkgd:
        img[bound_mask != 1] = 0 #1 if white_back else 0

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)

    near_all = np.zeros_like(ray_o[:,0])
    far_all = np.ones_like(ray_o[:,0])
    near_all[mask_at_box] = near 
    far_all[mask_at_box] = far 
    near = near_all
    far = far_all

    coord = np.zeros([len(ray_o), 2]).astype(np.int64)
    bkgd_msk = msk

    return img, ray_o, ray_d, near, far, coord, mask_at_box, bkgd_msk


class TikTokDatasetBatch(Dataset):
    def __init__(self, data_root=None, split='test', multi_person=True, num_instance=-1, poses_start=5, poses_interval=5, poses_num=10, image_scaling=0.5, white_back=False, sample_obs_view=True, fix_obs_view=False, resolution=None, saved_list_path='',
                 idx_start=0, idx_end=0, tgt_pose_index=0):
        super(TikTokDatasetBatch, self).__init__()
        self.data_root = data_root
        self.split = split
        self.image_scaling = image_scaling
        self.white_back = white_back
        self.sample_obs_view = sample_obs_view
        self.fix_obs_view = fix_obs_view

        self.poses_start = poses_start
        self.poses_interval = poses_interval 
        self.poses_num = poses_num 

        self.multi_person = multi_person
        self.num_instance = num_instance
        humans_data_root = os.path.dirname(data_root)
        

        self.cams_all = []        
        self.data_list = [data_root]

        # prepare t pose and vertex
        self.smpl_model = SMPL(sex='neutral', model_dir='') # model_dir directly defined in smpl_numpy.py
        self.big_pose_params = self.big_pose_params()
        t_vertices, _ = self.smpl_model(self.big_pose_params['poses'], self.big_pose_params['shapes'].reshape(-1))
        self.t_vertices = t_vertices.astype(np.float32)
        min_xyz = np.min(self.t_vertices, axis=0)
        max_xyz = np.max(self.t_vertices, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        min_xyz[2] -= 0.1
        max_xyz[2] += 0.1
        self.t_world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        self.num_freeviews = 20

        self.p = transforms.RandomResizedCrop(size=(512, 512), scale=(1.0, 1.0), ratio=(1.0, 1.0))

        self.pose_index = None

    def get_mask(self, mask_path):
        msk = imageio.imread(mask_path)
        msk[msk!=0]=255
        return msk

    
    def prepare_input(self, param):
        # body pose
        body_pose = param['smpls']['body_pose'][0]
        # pdb.set_trace()
        body_pose = np.array([Rotation.from_matrix(mat).as_rotvec() for mat in body_pose]).ravel()
        poses = np.zeros(72)
        poses[:3] = np.array(Rotation.from_matrix(param['smpls']['global_orient'][0][0]).as_rotvec()).astype(np.float32)
        poses[3:] = body_pose
        # pdb.set_trace()
        # betas
        betas = param['smpls']['betas']

        # verts
        xyz = param['verts'][0]
        R = np.eye(3).astype(np.float32) #param['smpls']['global_orient'][0][0]
        smpl_Th = np.zeros_like(param['cam_t'][0].reshape(-1, 3))

        del param
        
        vertices = xyz.astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        smpl_param = {
            'R' : R.astype(np.float32), # 3, 3
            'Th' : smpl_Th.astype(np.float32), # 1, 3
            'shapes' : betas.astype(np.float32), # 1 10
            'poses' : poses.astype(np.float32), # 72,
        }

        return world_bounds, vertices, smpl_param

    def big_pose_params(self):

        big_pose_params = {}
        # big_pose_params = copy.deepcopy(params)
        big_pose_params['R'] = np.eye(3).astype(np.float32)
        big_pose_params['Th'] = np.zeros((1,3)).astype(np.float32)
        big_pose_params['shapes'] = np.zeros((1,10)).astype(np.float32)
        big_pose_params['poses'] = np.zeros((1,72)).astype(np.float32)
        big_pose_params['poses'][0, 5] = 45/180*np.array(np.pi)
        big_pose_params['poses'][0, 8] = -45/180*np.array(np.pi)
        big_pose_params['poses'][0, 23] = -30/180*np.array(np.pi)
        big_pose_params['poses'][0, 26] = 30/180*np.array(np.pi)

        return big_pose_params

    def get_freeview_camera(self, frame_idx, total_frames, K, R, T, trans=None):
        # get free view camera based on its index
        RT = np.hstack((R, T))
        extri = np.vstack((RT, [0, 0, 0, 1]))
        R_updated, T_updated = rotate_camera_by_frame_idx(
                extrinsics=extri, 
                frame_idx=frame_idx,
                period=total_frames,
                trans=trans,
                )
        # K = self.train_camera['intrinsics'].copy()
        return K, R_updated, T_updated


    def __getitem__(self, index):
        instance_idx = index // self.num_freeviews

        view_index = index % self.num_freeviews

        img_all, ray_o_all, ray_d_all, near_all, far_all = [], [], [], [], []
        mask_at_box_all, bkgd_msk_all, mask_at_box_large_all = [], [], []
        obs_img_all, obs_K_all, obs_R_all, obs_T_all = [], [], [], []


        obs_pose_index = self.pose_index

        # Load image, mask, K, R, T in observation space
        obs_img_path = self.data_root
        obs_mask_path = os.path.join(
            os.path.dirname(obs_img_path),
            "groundsam_vis",
            os.path.basename(obs_img_path)+".mask.png"
        )

        obs_img = np.array(imageio.imread(obs_img_path).astype(np.float32) / 255.)
        obs_msk = np.array(self.get_mask(obs_mask_path)) / 255.
        obs_img[obs_msk == 0] = 1 if self.white_back else 0

        annot_path = os.path.join(
            os.path.dirname(obs_img_path),
            "smpl_results",
            os.path.basename(obs_img_path).split(".")[0]+".npy"
        )
        annot = np.load(annot_path, allow_pickle=True)[()]
        # pdb.set_trace()
        fl = annot['scaled_focal_length'][()]
        res_x, res_y = annot['render_res']
        cx, cy = res_x / 2, res_y / 2

        obs_K = np.array([
            [fl, 0, cx],
            [0, fl, cy],
            [0, 0, 1]
        ])

        obs_R = np.eye(3)
        obs_T = annot['cam_t'][0].reshape(-1, 1)
        # Prepare smpl in observation space
        _, obs_vertices, obs_params = self.prepare_input(annot) # vertices :  world coordinate

        
        obs_img = np.transpose(obs_img, (2,0,1))

        # obs view
        obs_img_all.append(obs_img)
        obs_K_all.append(obs_K)
        obs_R_all.append(obs_R)
        obs_T_all.append(obs_T)

        # target view
        pose_index = self.pose_index
        # Load image, mask, K, R, T, scale
        img_path = self.data_root

        img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
        msk = np.ones_like(obs_msk)
        img[msk == 0] = 1 if self.white_back else 0

        annot_path = os.path.join(
            os.path.dirname(obs_img_path),
            "smpl_results",
            os.path.basename(obs_img_path).split(".")[0]+".npy"
        )
        annot = np.load(annot_path, allow_pickle=True)[()]

        
        K, R, T = self.get_freeview_camera(
            frame_idx=view_index, total_frames=self.num_freeviews, K=obs_K, R=obs_R, T=obs_T
        )
        T = T.reshape(-1, 1)

        # Prepare the smpl input, including the current pose and canonical pose 
        world_bounds, vertices, params = self.prepare_input(annot) # vertices :  world coordinate

        
        # Sample rays in target space world coordinate
        img, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk = sample_ray_RenderPeople_batch(
                img, msk, K, R, T, world_bounds, 1.0)

        mask_at_box_large = mask_at_box

        
        img = np.transpose(img, (2,0,1))

        img_all.append(img)
        ray_o_all.append(ray_o)
        ray_d_all.append(ray_d)
        near_all.append(near)
        far_all.append(far)
        mask_at_box_all.append(mask_at_box)
        bkgd_msk_all.append(bkgd_msk)
        mask_at_box_large_all.append(mask_at_box_large)

        # target view
        img_all = np.stack(img_all, axis=0)
        ray_o_all = np.stack(ray_o_all, axis=0)
        ray_d_all = np.stack(ray_d_all, axis=0)
        near_all = np.stack(near_all, axis=0)[...,None]
        far_all = np.stack(far_all, axis=0)[...,None]
        mask_at_box_all = np.stack(mask_at_box_all, axis=0)
        bkgd_msk_all = np.stack(bkgd_msk_all, axis=0)
        mask_at_box_large_all = np.stack(mask_at_box_large_all, axis=0)

        # obs view 
        obs_img_all = np.stack(obs_img_all, axis=0)
        obs_K_all = np.stack(obs_K_all, axis=0)
        obs_R_all = np.stack(obs_R_all, axis=0)
        obs_T_all = np.stack(obs_T_all, axis=0)

        ret = {
            "instance_idx": instance_idx, 
            'pose_index': pose_index,
            "obs_pose_index": obs_pose_index,

            # canonical space
            't_params': self.big_pose_params,
            't_vertices': self.t_vertices,
            't_world_bounds': self.t_world_bounds,

            # target view
            "params": params, # smpl params including smpl global R, Th
            'vertices': vertices, # world vertices
            'img_all': img_all,
            'ray_o_all': ray_o_all,
            'ray_d_all': ray_d_all,
            'near_all': near_all,
            'far_all': far_all,
            'bkgd_msk_all': bkgd_msk_all,
            'mask_at_box_all': mask_at_box_all,
            'mask_at_box_large_all': mask_at_box_large_all,

            # obs view
            'obs_params': obs_params,
            'obs_vertices': obs_vertices,
            'obs_img_all': obs_img_all,
            'obs_K_all': obs_K_all,
            'obs_R_all': obs_R_all,
            'obs_T_all': obs_T_all,

        }

        return ret

    def __len__(self):
        return len(self.data_list) * self.num_freeviews