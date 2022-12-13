# SpatialBias
Spatial Bias for Attetion-free Non-local Neural Networks

![sb_overflow](https://user-images.githubusercontent.com/90232305/207276951-3e7f3f15-8aba-4778-b11c-2dbd1faf7210.jpg)


## Training Script

1. Clone this repo & install required library (assume that you have installed pytorch>1.5.0 and torchvision>0.7.0)

   ```bash
   git clone https://github.com/rwightman/pytorch-image-models
   pip3 install timm wandb
   ```

2. Create dataset folder link

   ```bash
   ln -s your/imagenet/folder your_imagenet_dir
   ```

3. Run `train.py` with one of below argument


4. Training Scripts(you should change your_imagenet_dir in the code below)
For the training script, refer to timm(https://github.com/rwightman/pytorch-image-models).
```bash
torchrun --nproc_per_node=4 --master_port=12348 train.py your_imagenet_dir --model sb_resnet50 --cuda 0,1,2,3 --aa rand-m7-mstd0.5-inc1 
          --mixup .1 --cutmix 1.0 --aug-repeats 3 --remode pixel --reprob 0.0 --crop-pct 0.95 --drop-path .05 --smoothing 0.0 
          --bce-loss --bce-target-thresh 0.2 --opt lamb --weight-decay .02 --sched cosine --epochs 300 --lr 3.5e-3 
          --warmup-lr 1e-4 -b 256 -j 16 --amp --channels-last --log-wandb --pin-mem
```

