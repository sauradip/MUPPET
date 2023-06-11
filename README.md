# STALE-FS

## Getting Started

### Requirements
- Python 3.7
- PyTorch == 1.9.0  **(Please make sure your pytorch version is atleast 1.8)**
- NVIDIA GPU
- Hugging-Face Transformers
- Detectron

### Environment Setup
It is suggested to create a Conda environment and install the following requirements
```shell script
pip3 install -r requirements.txt
```

### Extra Dependencies
We have used the implementation of [Maskformer](https://github.com/facebookresearch/MaskFormer) for Representation Masking. 
```shell script
git clone https://github.com/sauradip/STALE-FS.git
cd STALE-FS
git clone https://github.com/facebookresearch/MaskFormer
```
Follow the [Installation](https://github.com/facebookresearch/MaskFormer/blob/main/INSTALL.md) instructions to install Detectron and other modules within this same environment if possible. After this step, place the files in ``` /STALE-FS/extra_files ``` into ``` /STALE-FS/MaskFormer/mask_former/modeling/transformer/ ```. 

### Download Features ( For Frozen Encoder based Training )
Download the video features and update the Video paths/output paths in ``` config/anet.yaml ``` file. For now ActivityNetv1.3 dataset config is available. We are planning to release the code for THUMOS14 dataset soon. 

| Dataset | Feature Backbone | Pre-Training | Link | 
|:---:|:---:|:---:|:---:|
| ActivityNet | ViT-B/16-CLIP | CLIP | [Google Drive](https://drive.google.com/drive/folders/1OFyU7V-VPHYOkTfXTQR-XxLYO-rSgL_i?usp=sharing) |
| THUMOS | ViT-B/16-CLIP | CLIP | [Google Drive](https://drive.google.com/drive/folders/16eUrTrF8-S5ncb5psIN7ikP9GweAIP_t?usp=sharing) |
| ActivityNet | I3D | Kinetics-400 | [Google Drive](https://drive.google.com/drive/folders/1B1srfie2UWKwaC4-7bo6UItmJoESCUq3?usp=sharing) |
| THUMOS | I3D | Kinetics-400 | [Google Drive](https://drive.google.com/drive/folders/1C4YG01X9IIT1a568wMM8fgm4k4xTC2EQ?usp=sharing) |

### Setup Pre-Training Data
We followed the pre-processing of [AFSD](https://github.com/TencentYoutuResearch/ActionDetection-AFSD) to pre-process the video data. For the Adaptformer we selected Kinetics pre-trained weights available at this [link](https://github.com/ShoufaChen/AdaptFormer/blob/main/PRETRAIN.md). We use this as initialized weights and only learn few parameters in adapter for the downstream few-shot episodes. 

### Training Splits
Currently we support the training-splits provided by [QAT](https://github.com/sauradip/fewshotQAT) as there is no publicly available split apart from random split. 

### Model Training 
The model training happens in 2 phases. But first set up all weights and paths in ``` config/anet.yaml ``` file.

#### (1) Base Split Training
To pre-train STALE-FS on base-class split , first set the parameter ``` ['fewshot']['mode'] = 0 ``` , ``` ['fewshot']['shot'] = 1/2/3/4/5 ```, ``` ['fewshot']['nway'] = 1/2/3/4/5 ``` in ``` config/anet.yaml ``` file. Then run the following command.

```shell script
python stale_pretrain_base.py
```
After Training with Base split , now our model is ready to transfoer knowledge to novel classes. But still we need to adapt the decoder head using few-support (few-shot) samples. For this we need another stage training. 

#### (2) Meta Training
To pre-train STALE-FS on novel-class split , first set the parameter ``` ['fewshot']['mode'] = 2 ``` , ``` ['fewshot']['shot'] = 1/2/3/4/5 ```, ``` ['fewshot']['nway'] = 1/2/3/4/5 ``` in ``` config/anet.yaml ``` file. Then run the following command.

```shell script
python stale_inference_meta_pretrain.py
```
During this stage the Adapter should learn the parameters specific to training episodes. 

### Model Inference
The checkpoints will be automatically saved to output directly if properly set-up in ``` config/anet.yaml ``` file. Additionally, set the parameter ``` ['fewshot']['mode'] = 3 ``` , ``` ['fewshot']['shot'] = 1/2/3/4/5 ```, ``` ['fewshot']['nway'] = 1/2/3/4/5 ``` in ``` config/anet.yaml ``` file. Then run the following command.
```shell script
python stale_inference_meta_pretrain.py
```
Same .py file as meta-train but the few-shot mode is changed. This step shows a lot of variation as random classes are picked up as query and intra-class variation causes mAP to vary. SO a high number of few-shot episodes is/should be kept. This command saves the video output and post-processes. 

### Model Evaluation
To evaluate our STALE-FS model run the following command. 
```shell script
python eval.py
```

### Acknowledgement
Our source code is based on implementations of [DenseCLIP](https://github.com/raoyongming/DenseCLIP), [MaskFormer](https://github.com/facebookresearch/MaskFormer) and [CoOP](https://github.com/kaiyangzhou/coop). We thank the authors for open-sourcing their code. 

