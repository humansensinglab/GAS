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
- [x] Inference code for novel view synthesis
- [x] GAS checkpoints release on Hugging Face 
- [ ] Inference code for novel pose synthesis
- [ ] Inference code for synchronized multi-view video generation

## Requirements
* An NVIDIA GPU with CUDA support is required. 
  * We have tested on a single A4500 and A100 GPU.
  * The minimum GPU memory required for inference is 20GB.
* Operating system: Linux
## Installation
Instructions can be found in [INSTALL.md](docs/INSTALL.md).


## Inference
### Novel view synthesis
```
bash demo_scripts/nv.sh
```

### Novel pose synthesis
TBD

### Multi-view video generation
TBD

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
