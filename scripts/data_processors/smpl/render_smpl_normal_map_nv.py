import pip

pip.main(["install", "tqdm"])

import bpy
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
from contextlib import contextmanager
import pathlib

pip.main(["install", "opencv-python"])
import cv2

this_script_path = pathlib.Path(__file__).parent.resolve()
W_FACE_AND_COLOR_FILE = this_script_path / "blend" / "smpl_mesh_info.npy"

FORMAT_LDR = "PNG"
COLOR_DEPTH_LDR = 8
SAMPLES = 1
COLOR_MODE = "RGB"


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

    return rot_camrot.T, -rot_camrot.T.dot(rot_campos) # R, T


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


def get_freeview_camera(frame_idx, total_frames, K, R, T, trans=None):
    # get free view camera based on its index
    RT = np.hstack((R, T)) # (3, 3) (3, 1)
    extri = np.vstack((RT, [0, 0, 0, 1]))
    R_updated, T_updated = rotate_camera_by_frame_idx(
            extrinsics=extri, 
            frame_idx=frame_idx,
            period=total_frames,
            trans=trans,
            )
    return K, R_updated, T_updated





def setup_device(use_id):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for i, device in enumerate(
        bpy.context.preferences.addons["cycles"].preferences.devices
    ):
        if i == use_id or "CPU" in device["name"]:
            device["use"] = True  # Using all devices, include GPU and CPU
        else:
            device["use"] = False  # Using all devices, include GPU and CPU
        print(device["name"], "USE:", bool(device["use"]))

class SingleDataset:
    def __init__(self, smpl_folder, reference_image_path, smpl_suffixes=[".npy", ".npz"]):
        # smpl_suffixes = 'npy' # NOTE only allow npy files here: exclude smpls_group_smooth.npz and 
        reference_image_name = os.path.basename(reference_image_path).split(".")[0]
        self.smpl_folder = Path(smpl_folder)
        self.out_folder = Path("reference_images_freeview_smpl") / reference_image_name
        self.smpl_paths = []
        self.bboxes = []
        self.valid_index = []
        
        
        # import pdb; pdb.set_trace()
        self.smpl_paths = sorted(
            [
                path for i in smpl_suffixes
                for path in (self.smpl_folder / "smpl_results").glob(reference_image_name + i)
            ]
        )
        self.output_paths = [self.out_folder for smpl_path in self.smpl_paths]
        # Skip finished smpl_path. Enable it if want to continue processing only remaining imgs.
        smpl_fns = [
            os.path.splitext(os.path.basename(smpl_path))[0]
            for smpl_path in self.smpl_paths
        ]  
        imgs_output_path = [
            os.path.join(str(self.output_paths[i]), "visualized_imgs", f"{smpl_fn}.png")
            for i, smpl_fn in enumerate(smpl_fns)
        ] 
        
        imgs_already_exist = [False for _ in imgs_output_path] # overwrite all previous renderings
        imgs_index_to_inference = np.where(np.array(imgs_already_exist) == False)[0] # only render the images that do not exist
        smpl_paths_copy = list(self.smpl_paths)
        output_paths_copy = list(self.output_paths)
        self.smpl_paths = [smpl_paths_copy[img_index] for img_index in imgs_index_to_inference]
        self.output_paths = [output_paths_copy[img_index] for img_index in imgs_index_to_inference]
        print("output paths: ", self.output_paths)
        print(f"finish loading {imgs_index_to_inference.shape[0]} frames data, \
              skip {len(smpl_fns) - imgs_index_to_inference.shape[0]} existing images")


def load_smpl(smpl_path):
    return np.load(smpl_path, allow_pickle=True).item()


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    Redirects stdout to a specified file.

    Usage:
    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    # Save a copy of the original stdout file descriptor
    original_stdout_fd = os.dup(fd)

    # Redirect stdout to the specified file
    with open(to, 'w') as file:
        os.dup2(file.fileno(), fd)

    try:
        yield
    finally:
        # Restore the original stdout
        os.dup2(original_stdout_fd, fd)
        os.close(original_stdout_fd)

def rendering_pipeline(dataset, ref_img_path):
    scene = bpy.context.scene
    scene.render.image_settings.file_format = str(FORMAT_LDR)
    scene.render.image_settings.color_depth = str(COLOR_DEPTH_LDR)
    scene.render.image_settings.color_mode = str(COLOR_MODE)
    scene.render.resolution_percentage = 100
    scene.render.use_persistent_data = True
    scene.cycles.use_denoising = False

    camera = bpy.data.objects["Camera"]
    camera.data.clip_start = 0.05
    camera.data.clip_end = 1e12
    camera.data.cycles.samples = SAMPLES
    scene.cycles.samples = SAMPLES
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    render_layers = bpy.context.scene.view_layers
    if "mesh_collection" not in bpy.data.collections.keys():
        mesh_collection = bpy.data.collections.new("mesh_collection")
        bpy.context.scene.collection.children.link(mesh_collection)
    mesh_collection = bpy.data.collections.get("mesh_collection")

    mat_normal = bpy.data.materials.get("Normal")

    result_dict = np.load(W_FACE_AND_COLOR_FILE, allow_pickle=True).item()
    faces = result_dict["faces"]

    result_dict_list = []

    processed = []
    for path in tqdm(dataset.smpl_paths, total=len(dataset.smpl_paths), desc="Loading smpls into RAM"):
        result = load_smpl(path)
        processed.append(result)

    for smpl in tqdm(processed, total=len(dataset.smpl_paths), desc="Loading smpls into RAM"):
        result_dict_list.append(smpl)

    for smpl_path, output_path, result_dict in tqdm(
        zip(dataset.smpl_paths, dataset.output_paths, result_dict_list),
        total=len(dataset.output_paths),
        desc="Rendering Images",
        miniters=10,
    ):
        render_path = output_path
        smpl_fn, _ = os.path.splitext(os.path.basename(smpl_path))
        smpl_fn = smpl_fn.split(".")[0]
        base_dirname = os.path.dirname(os.path.dirname(smpl_path))

        base_R = np.eye(3)
        base_T = result_dict["cam_t"][0].reshape(-1, 1) 
        base_K = None
        num_freeviews = 20 

        for view_name in range(num_freeviews):
            view_name = str(view_name).zfill(2)
            if view_name == '00':
                ref_img_path = os.path.join(base_dirname, smpl_fn+'.png')

                ori_img = bpy.data.images.load(ref_img_path)
                ori_img.name = "ori_img.png"
                ori_img.colorspace_settings.name = "Raw"
                bpy.data.scenes["Scene"].node_tree.nodes["Image"].image = ori_img

            _, R, T = get_freeview_camera(int(view_name), num_freeviews, base_K, base_R, base_T)


            verts = result_dict["verts"][0] @ R.T + T.squeeze()
            img_size = result_dict["render_res"].astype(np.int32)
            camera.data.sensor_width = img_size.max()
            camera.data.lens = result_dict["scaled_focal_length"]
            scene.render.resolution_x = img_size[0]
            scene.render.resolution_y = img_size[1]

            # make mesh
            new_mesh = bpy.data.meshes.new("smpl_mesh")
            new_mesh.from_pydata(verts, edges=[], faces=faces)
            # make object from mesh
            new_object = bpy.data.objects.new("new_object", new_mesh)
            # make collection
            new_object.rotation_euler[0] = -np.pi / 2

            for f in new_object.data.polygons:
                f.use_smooth = True

            # add object to scene collection
            mesh_collection.objects.link(new_object)

            new_object.data.materials.append(mat_normal)

            scene.view_settings.view_transform = "Raw"
            bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
            output_name = f"{smpl_fn}.png"

            normal_path = os.path.join(render_path, "normal", view_name, output_name)

            scene.render.filepath = normal_path
            for layer in render_layers:
                # some condition
                layer.use = layer.name == "ViewLayer"
            bpy.context.scene.render.film_transparent = True
            with stdout_redirected():
                bpy.ops.render.render(write_still=True)

            bpy.data.objects.remove(new_object, do_unlink=True)
            bpy.data.meshes.remove(new_mesh)
            if view_name == '00':
                bpy.data.images.remove(ori_img)
                
            del new_mesh
            if view_name == '00':
                del ori_img
            del new_object
    del result_dict_list


if __name__ == "__main__":
    import sys

    argv = sys.argv
    print(f"Rendering:")
    try:
        argv.index("--device")
    except:
        print("Use Only CPU for Rendering")
    else:
        setup_device(int(argv[argv.index("--device") + 1]))

    smpl_folder = argv[argv.index("--driving_path") + 1]
    ref_img_path = argv[argv.index("--reference_path") + 1]
    dataset = SingleDataset(smpl_folder, ref_img_path)
    rendering_pipeline(dataset, ref_img_path)
