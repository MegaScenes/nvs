# MegaScenes: Scene-Level View Synthesis at Scale

[**Paper**](https://megascenes.github.io/MegaScenes_paper_v1.pdf) | [**Arxiv**](https://arxiv.org/abs/2406.11819) | [**Project Page**](https://megascenes.github.io) <br>


This repository contains the official implementation of single-image novel view synthesis (NVS) from the project **MegaScenes: Scene-Level View Synthesis at Scale**. Details on the dataset can be found [here](https://github.com/MegaScenes/dataset).

If you find our code or paper useful, please consider citing
```bibtex
@inproceedings{
      tung2024megascenes,
      title={MegaScenes: Scene-Level View Synthesis at Scale}, 
      author={Tung, Joseph and Chou, Gene and Cai, Ruojin and Yang, Guandao and Zhang, Kai and Wetzstein, Gordon and Hariharan, Bharath and Snavely, Noah},
      booktitle={ECCV},
      year={2024}
    }
```

## Installation
We recommend creating a [conda](https://www.anaconda.com/) environment then installing the required packages using the following commands:

```
conda create -n megascenes python=3.8 pip --yes
conda activate megascenes
bash setup_env.sh
```

Additionally, install Depth Anything following the instructions from the official [repository](https://github.com/LiheYoung/Depth-Anything). This will be required for inference. 

## Downloading Pretrained Models 
We provide two checkpoints in the MegaScenes AWS bucket. Download the folder `s3://megascenes/nvs_checkpoints/warp_plus_pose/iter_112000/` to the directory `configs/warp_plus_pose/iter_112000/`. This model is conditioned on warped images and poses as described in the paper. Download the folder `s3://megascenes/nvs_checkpoints/zeronvs_finetune/iter_90000/` to the directory `configs/zeronvs_finetune/iter_90000/`. This checkpoint is ZeroNVS finetuned on MegaScenes. For comparison, also download the original ZeroNVS [checkpoint](https://drive.google.com/file/d/17WEMfs2HABJcdf4JmuIM3ti0uz37lSZg/view) to the directory `configs/zeronvs_original/iter_0/zeronvs.ckpt`.

## Inference 
The following commands create videos based on two pre-defined camera paths. `-i` points to the path of the reference image and `-s` is the output path. <br>
The generated `.gif` files will be located at `qual_eval/warp_plus_pose/audley/orbit/videos/best.gif` and `.../spiral/videos/best.gif`. The warped images at each camera location will be located at `qual_eval/warp_plus_pose/audley/orbit/warped/warps.gif` and `.../spiral/warped/warps.gif`. <br>
Adjust the batch size as needed. 

### Model conditioned on warped images and poses
```
python video_script.py -e configs/warp_plus_pose/ -r 112000 -i data/examples/audley_end_house.jpg -s qual_eval/warp_plus_pose/audley
```

### Model conditioned on poses (i.e. finetuning ZeroNVS)
```
python video_script.py -e configs/zeronvs_finetune/ -r 90000 -i data/examples/audley_end_house.jpg -s qual_eval/zeronvs_finetune/audley -z
``` 

### Original ZeroNVS checkpoint 
```
python video_script.py -e configs/zeronvs_original/ -r 0 -i data/examples/audley_end_house.jpg -s qual_eval/zeronvs_original/audley -z --ckpt_file
```

## Dataset
The MegaScenes dataset is hosted on AWS. Documentation can be found [here](https://github.com/MegaScenes/dataset). Training NVS requires image pairs and their camera parameters and warpings. We provide the filtered image pairs and camera parameters in `s3://megascenes/nvs_checkpoints/splits/`. Download the folder to `data/splits/`. <br> 
Each `.pkl` file is a list of lists with the format <br>
`[img 1, img2, {img 1 extrinsics, img 1 intrinsics}, {img 2 extrinsics, img 2 intrinsics}, scale (of img 1's translation vector based on 20th quantile of depth)]`. <br>
See `dataloader/paired_dataset.py` for details.
We recommend preprocessing warped images. We provide code to warp a reference image to a target pose given its depth map and camera parameters. 
```
from dataloader.util_3dphoto import unproject_depth, render_view
mesh = unproject_depth('mesh_path.ply', img, depthmap, intrinsics, c2w_original_pose, scale_factor=1.0, add_faces=True, prune_edge_faces=True)
warped_image, _ = render_view(h, w, intrinsics, c2w_target_pose, mesh)
```
We currently do not provide the aligned depth maps and warped images.
<br>


## Training
```
accelerate launch --config_file acc_configs/{number_of_gpus}.yaml train.py -e configs/warp_and_pose/ -b {batch_size} -w {workers}  
```
We use a batch size of 88 on an A6000 with 49G of vram. 


## Testing
```
python test.py -e configs/warp_plus_pose -r 112000 -s warp_plus_pose_evaluation -b {batch_size} -w {workers} --save_generations True --save_data
python test.py -e configs/zeronvs_finetune -r 90000 -s zeronvs_evaluation -b {batch_size} -w {workers} --save_generations True 
```
Generated images and metrics are saved to `quant_eval/warp_plus_pose_evaluation`. `-r` loads the saved checkpoint. The warped images also should be prepared in advance for calculating metrics. 


## References
We adapt code from <br>
Zero-1-to-3 https://zero123.cs.columbia.edu/ <br>
ZeroNVS https://kylesargent.github.io/zeronvs/
