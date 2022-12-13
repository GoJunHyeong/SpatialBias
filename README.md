# Spatial Bias for Attetion-free Non-local Neural Networks

![sb_overflow](https://user-images.githubusercontent.com/90232305/207277348-4679dbe5-e4b5-45c2-a167-b0acd14b7ebf.jpg)

## Abstract
In this paper, we introduce the spatial bias to learn global knowledge without self-attention in convolutional neural networks. Owing to the limited receptive field, conventional convolutional neural networks suffer from learning long-range dependencies. Non-local neural networks have struggled to learn global knowledge, but unavoidably have too heavy a network design due to the self-attention operation. Therefore, we propose a fast and lightweight spatial bias that efficiently encodes global knowledge without self-attention on convolutional neural networks. Spatial bias is stacked on the feature map and convolved together to adjust the spatial structure of the convolutional features. Therefore, we learn the global knowledge on the convolution layer directly with very few additional resources. Our method is very fast and lightweight due to the attention-free non-local method while improving the performance of neural networks considerably. Compared to non-local neural networks, the spatial bias use about $\times 10$ times fewer parameters while achieving comparable performance with $1.6\sim3.3$ times more throughput on a very little budget. Furthermore, the spatial bias can be used with conventional non-local neural networks to further improve the performance of the backbone model.
We show that the spatial bias achieves competitive performance that improves the classification accuracy by $+0.79\%$ and $+1.5\%$ on ImageNet-1K and cifar100 datasets. Additionally, we validate our method on the MS-COCO and ADE20K datasets for downstream tasks involving object detection and semantic segmentation.

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

