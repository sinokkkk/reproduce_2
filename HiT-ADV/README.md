# Hide in Thicket: Generating Imperceptible and Rational Adversarial Perturbations on 3D Point Clouds  (CVPR 2024)

This work corresponds to the following paper: https://arxiv.org/abs/2403.05247.

This repo is based on [https://github.com/code-roamer/AOF.git] and [https://github.com/shikiw/SI-Adv.git]  to perform adversarial attack. 

![Framework](https://github.com/TRLou/HiT-ADV/assets/133848600/6e4f5b82-63fe-4084-a18c-a34e506d0e30)

```
@article{lou2024HiT-ADV,
  title={Hide in Thicket: Generating Imperceptible and Rational Adversarial Perturbations on 3D Point Clouds},
  author={Lou, Tianrui and Jia, Xiaojun and Gu, Jindong and Liu, Li and Liang, Siyuan and He, Bangyan and Cao, Xiaochun},
  journal={arXiv preprint arXiv:2403.05247},
  year={2024}
}
```
# Get Started
Step 1. Create a conda environment or use your existing one.
```
conda create --name hitadv python=3.8 -y
conda activate hitadv
```

For AutoDL, use a PyTorch/CUDA image such as PyTorch 1.11.0 + CUDA 11.3, then install dependencies in this order:

```
conda create -n hitadv python=3.8 -y
conda activate hitadv

conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt

pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2"
cd pointnet2_ops_lib
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" pip install -e .
cd ..
```

If you only need `eval.py`, Mayavi is not required. To run `visual.py` on a headless AutoDL instance, install visualization dependencies separately:

```
pip install mayavi==4.8.1 PyQt5
```

Step 2. Prepare datasets and pretrained models.
Download from Baidu Yun：https://pan.baidu.com/s/1SL5-TuT9n74x5mADSM2E9g
(password:eaic)

Put them in paths like:

```
HiT-ADV/
  Checkpoint/PN_NT.checkpoint
PC_Dataset/
  modelnet40_normal_resampled/
  shapenetcore_partanno_segmentation_benchmark_v0_normal/
```

Step 3. Evaluating：
```
python eval.py --data_root ../PC_Dataset --checkpoint Checkpoint/PN_NT.checkpoint --gpu 0
```
Visualizing:
```
python visual.py --data_root ../PC_Dataset --checkpoint Checkpoint/PN_NT.checkpoint --gpu 0
```


