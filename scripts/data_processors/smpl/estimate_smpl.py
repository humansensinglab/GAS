import cv2
from pathlib import Path
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import platform
import pyrender
import sys
sys.path.append('4D-Humans')
sys.path.append('./')
from scripts.pretrained_models import (
    DETECTRON2_MODEL_PATH,
    HMR2_DEFAULT_CKPT,
)

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

from hmr2.models import load_hmr2
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from scripts.data_processors.smpl.smpl_visualizer import SemanticRenderer

# For Windows, remove PYOPENGL_PLATFORM to enable default rendering backend
sys_name = platform.system()
if sys_name == "Windows":
    os.environ.pop("PYOPENGL_PLATFORM")

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)




def predict_smpl(batch, model, model_cfg, figure_scale=None):
    all_verts = []
    all_cam_t = []
    with torch.no_grad():
        out = model(batch)

    pred_cam = out["pred_cam"]
    pred_smpl_parameter = out["pred_smpl_params"]
    if figure_scale is not None:
        pred_smpl_parameter['betas'][0][1] = float(figure_scale)
    smpl_output = model.smpl(
        **{k: v.float() for k, v in pred_smpl_parameter.items()},
        pose2rot=False,
    )
    pred_vertices = smpl_output.vertices
    out["pred_vertices"] = pred_vertices.reshape(
        batch["img"].shape[0], -1, 3
    )

    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()

    scaled_focal_length = (
        model_cfg.EXTRA.FOCAL_LENGTH
        / model_cfg.MODEL.IMAGE_SIZE
        * img_size.max()
    )
    pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()
    # Render the result
    batch_size = batch["img"].shape[0]
    for n in range(batch_size):
        # Add all verts and cams to list
        verts = out["pred_vertices"][n].detach().cpu().numpy()
        cam_t = pred_cam_t_full[n]
        all_verts.append(verts)
        all_cam_t.append(cam_t)

    misc_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=scaled_focal_length,
    )
    
    smpl_outs = {
        k: v.detach().cpu().numpy() for k, v in pred_smpl_parameter.items()
    }

    results_dict_for_rendering = {
        "verts": all_verts,
        "cam_t": all_cam_t,
        "render_res": img_size[n].cpu().numpy(),
        "smpls": smpl_outs,
        "scaled_focal_length": scaled_focal_length.cpu().numpy(),
    }
    return results_dict_for_rendering, misc_args

def load_image(img_cv2, detector):
    # Detect humans in image
    det_out = detector(img_cv2)
    det_instances = det_out["instances"]
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    # Run HMR2.0 on all detected humans
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
    return torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=0
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference SMPL with 4D-Humans")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--reference_imgs_folder",
        type=str,
        default="",
        help="Folder path to reference imgs",
    )
    parser.add_argument(
        "--driving_video_path",
        type=str,
        default="",
        help="Folder path to driving videos",
    )
    parser.add_argument(
        "--figure_scale",
        type=int,
        default=None,
        help="Adjust the figure scale to better fit extreme shape",
    )

    parser.add_argument(
        "--reference_img_path",
        type=str,
        default="",
        help="Folder path to reference imgs",
    )

    args = parser.parse_args()


    if args.driving_video_path: 
        # include all videos
        # driving_videos_paths = [args.driving_video_path]
        driving_videos_paths = [path
        for path in os.listdir(args.driving_video_path)
        ]
        driving_videos_paths = sorted(driving_videos_paths)

    model, model_cfg = load_hmr2(HMR2_DEFAULT_CKPT)

    # Load detector
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hmr2

    cfg_path = (
        Path(hmr2.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    )
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = str(DETECTRON2_MODEL_PATH)
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    model = model.to(args.device)
    detector.model.to(args.device)
    # This PyRender is only used for visualizing, we use Blender after to render different conditions.
    renderer = SemanticRenderer(
        model_cfg,
        faces=model.smpl.faces,
        lbs=model.smpl.lbs_weights,
        viewport_size=(720, 720),
    )

    reference_img_paths = [args.reference_img_path]
    args.reference_imgs_folder = os.path.dirname(reference_img_paths[0])
    for img_path in tqdm(reference_img_paths, desc="Processing Reference Images:"):
        img_cv2 = cv2.imread(args.reference_img_path)

        renderer.renderer.delete()
        renderer.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_cv2.shape[:2][::-1][0],
            viewport_height=img_cv2.shape[:2][::-1][1],
            point_size=1.0,
        )
        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        dataloader = load_image(img_cv2, detector)

        for batch in dataloader:
            batch = recursive_to(batch, args.device)
            results_dict_for_rendering, misc_args = predict_smpl(batch, model, model_cfg, args.figure_scale)

            rendering_results = renderer.render_all_multiple(
                results_dict_for_rendering["verts"], 
                cam_t=results_dict_for_rendering["cam_t"], 
                render_res=results_dict_for_rendering["render_res"], **misc_args
            )
            # Overlay image
            valid_mask = rendering_results["Image"][:, :, -1][:, :, np.newaxis]
            cam_view = (
                valid_mask * rendering_results["Image"][:, :, [2, 1, 0]]
                + (1 - valid_mask) * img_cv2.astype(np.float32)[:, :, ::-1] / 255
            )
            os.makedirs(
                os.path.join(args.reference_imgs_folder, "smpl_results",),
                exist_ok=True
            )
            np.save(
                str(os.path.join(args.reference_imgs_folder, "smpl_results", f"{img_fn}.npy")),
                results_dict_for_rendering)

  