# SpatialBias
Attention-free Non-local Neural Network

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


4. Training (you should change your_imagenet_dir in the code below)
```bash
CUDA_VISIBLE_DEVICES=0,3,6,9 python3 -m torch.distributed.launch 
          --nproc_per_node=4 --master_port=12346 train.py your_imagenet_dir --model spatial_bias --aa rand-m7-mstd0.5-inc1 
          --mixup .1 --cutmix 1.0 --aug-repeats 3 --remode pixel --reprob 0.0 --crop-pct 0.95 --drop-path .05 --smoothing 0.0 
          --bce-loss --bce-target-thresh 0.2 --opt lamb --weight-decay .02 --sched cosine --epochs 300 --lr 3.5e-3 
          --warmup-lr 1e-4 -b 256 -j 16 --amp --channels-last --log-wandb --pin-mem
```


## Hyper parameters
In eca_like_nlc.py, There are two hyper parameters, gc_group & gc_divide_factor

1. `gc_group = [bool, bool, bool, bool]`
This determines whether to create a global context for each layer of the network.
--> 1 for True, 0 for False.

2. `gc_divide_factor = 64` (default)
divide factor of channel reduction before global convolution, and also the number of global contexts to be added. (16,32,64...)
So, When 64 is selected, the input channel 64 is dimensionally reduced to 1 to generate one global context.
