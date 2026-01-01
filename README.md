# GAS: Generative Avatar Synthesis from a Single Image
  <p align="center">
  <strong>ICCV 2025</strong>
  </p>
  <p align="center">
    <a href="https://humansensinglab.github.io/GAS/"><strong>Project Page</strong></a>
    |
    <a href="https://arxiv.org/abs/2502.06957"><strong>Paper</strong></a>

  </p> 

![Teaser figure](assets/teaser.png)
## Code release plan
- [x] Checkpoints release on Hugging Face 
- [x] Inference code for novel view synthesis
- [x] Inference code for novel pose synthesis

## Requirements
* An NVIDIA GPU with CUDA support is required. 
  * We have tested on a single A4500 and A100 GPU.
  * The minimum GPU memory required for inference is 20GB.
* Operating system: Linux
## Installation
Instructions can be found in [INSTALL.md](docs/INSTALL.md).


## Inference
We support both view and pose synthesis tasks. For each task, we present a quick demo and guidelines for running with custom data.

### Novel view synthesis
#### Quick demo
```
bash demo_scripts/nv.sh
```
#### Run with custom data
If you want to use your own image for inference, please create a new folder (e.g., `assets/demo_images` in our case) and move your own image into this folder. Note that all the intermediate renderings and the final result will be generated inside this folder. 

To run with custom data, 
- `demo_scripts/nv.sh`: Replace the `IMAGE_PATH` in L3 with your own image path. 
- `configs/inference/novel_views.yaml`: Replace `data.obs_img_path` in L10 with your own image path


### Novel pose synthesis
#### Quick demo
```
bash demo_scripts/np.sh
```
In this quick demo, we do self-reenactment. If you want to do cross-reenactment, you need to transfer the SMPL shape of the source image to the target SMPL parameters. 

#### Run with custom data
If you want to use your own source image / driving video for inference, please create a new folder (e.g., `assets/demo_videos` in our case) and move your own video into this folder. Note that all the intermediate renderings and the final result will be generated inside this folder. 

For the configs, you need to do the following modifications:

- `demo_scripts/np.sh`: Make your driving video path be `VIDEO_PATH` in L4; source image path be `IMAGE_PATH` in L9

- `configs/inference/novel_pose.yaml`: L9 `data.obs_img_path` be your source image path; L10  `data.driving_video_dir` be the parent folder that contains the driving video


## Citation

If you find this code useful for your research, please cite it using the following BibTeX entry.

```
@article{lu2025gas,
  title={Gas: Generative avatar synthesis from a single image},
  author={Lu, Yixing and Dong, Junting and Kwon, Youngjoong and Zhao, Qin and Dai, Bo and De la Torre, Fernando},
  journal={arXiv preprint arXiv:2502.06957},
  year={2025}
}
```
## Acknowledgements
This project builds on components that were implemented and released by [SHERF](https://github.com/skhu101/SHERF?tab=readme-ov-file), [Champ](https://github.com/fudan-generative-vision/champ), [MimicMotion](https://github.com/tencent/MimicMotion). We thank the authors for releasing the code. Please also consider citing these works if you find them helpful.

## Disclaimer
The purpose of this work is for research only. All the copyrights of the demo images and videos are from community users. If there is any infringement or offense, please get in touch with us (yixing@cs.unc.edu), and we will delete them in time.
