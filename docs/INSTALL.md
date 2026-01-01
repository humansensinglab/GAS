### 1. Set up the environment
```
# clone the project
git clone https://github.com/humansensinglab/GAS.git
cd GAS

# set up conda environment
conda create -n GAS python=3.11 -y
conda activate GAS 

# make sure that the pytorch cuda is consistent with the system cuda
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# install dependencies
pip install -r requirements.txt
```

### 2. Prepare for human foreground segmentation
```
cd modules/grounded-sam/GroundingDINO
pip install -e .

cd ../segment_anything
pip install -e .

cd ../../..
```

### 3. Prepare for SMPL estimation & rendering
We use SMPL & Rendering scripts from Champ to estimate SMPL and render normal maps. Please refer to [their doc](https://github.com/fudan-generative-vision/champ/blob/master/docs/data_process.md) for details. You may make minor changes according to your device setup.

1. Install Blender
```
# Download blender
wget https://download.blender.org/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz

# unzip this file
...

# Optionally create a soft link for easier calls
...
```
2. Install 4D-Humans 
```
# Install 4D-Humans
pip install git+https://github.com/shubham-goel/4D-Humans

# Download its checkpoints
python -m scripts.pretrained_models.download --hmr2
```

3. Install detectron2
```
# Download checkpoints
python -m scripts.pretrained_models.download --detectron2
```

### 4. Register SMPL Model
Register and download SMPL NEUTRAL model [here](https://smpl.is.tue.mpg.de/). 

- Copy it to your 4D-Humans installation directory
```
cp basicModel_neutral_lbs_10_207_0_v1.0.0.pkl /path/to/4DHumans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```
- Copy it to your sherf directory.

```
cp basicModel_neutral_lbs_10_207_0_v1.0.0.pkl modules/sherf/assets/SMPL_NEUTRAL.pkl

# file structure should be as below
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
