### 1. Set up the environment
```
# clone the project
git clone https://github.com/humansensinglab/GAS.git
cd GAS

# set up conda environment
conda create -n GAS python=3.11
conda activate GAS 

# make sure that the pytorch cuda is consistent with the system cuda
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# install dependencies
pip install -r requirements.txt
```

### 2. Prepare for human foreground segmentation
```
cd annotator/grounded-sam/GroundingDINO
pip install -e .

cd ../segment_anything
pip install -e .

cd ../../..
```

### 3. Prepare for SMPL estimation & rendering
We use [SMPL & Rendering scripts](https://github.com/fudan-generative-vision/champ/blob/master/docs/data_process.md) to estimate SMPL and render normal maps. 

<details>
<summary>Checklist</summary>

- Install blender
- Install 4D-Humans under this project directory; 
use `python -m scripts.pretrained_models.download --hmr2` to download the checkpoints
- Install detectron2; use `python -m scripts.pretrained_models.download --detectron2` to download the checkpoints
</details>

### 4. Prepare for human NeRF rendering
Register and download SMPL NEUTRAL model [here](https://smpl.is.tue.mpg.de/). See the folder structure below. 
```
./
├── modules/
    ├── sherf/
        ├── ...
        └── assets/
            ├── SMPL_NEUTRAL.pkl
```


### 5. Download pretrained model weights
1. Download Grounded-SAM checkpoints

    ```
    mkdir -p pretrained_models/Grounded_SAM
    cd pretrained_models/Grounded_SAM
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    ```

2. Download Stable Video Diffusion checkpoints

    Download SVD model from [stabilityai/stable-video-diffusion-img2vid-xt-1-1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1).

3. Download GAS model weights 

    Download GAS and SHERF pretrained models from [here](https://huggingface.co/yixinglu/GAS). 

4. Model checkpoints folder structure 
    ```
    ./
    ├── pretrained_models/
        ├── detectron2/
            ├── model_final_f05665.pkl
        └── GAS/
            ├── nerf_net.pth
            └── pose_net.pth
            └── unet.pth
        └── Grounded_SAM/
            ├── groundingdino_swint_ogc.pth
            └── sam_vit_h_4b8939.pth
        └── sherf/
            ├── network-snapshot.pkl
        └── stable-video-diffusion-img2vid-xt-1-1/
            ├── ...
    ```
