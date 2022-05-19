# Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch.nn as nn
import math
import torch
import numpy as np

from timm.models.registry import register_model

@register_model
def spatial_bias(pretrained=False, **kwargs):
    return ResNet('imagenet',depth=50, num_classes=1000, bottleneck=True, gc_group=[True,True,True,False])

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, gc_group=False, resolution=-1):
        super(Bottleneck, self).__init__()

        if gc_group:
            gc_in_plane = 5

            upsample_factor = resolution / 10
            self.conv_gc_reduction = nn.Conv2d(planes, gc_in_plane, kernel_size=1, bias=False, stride=stride)
            self.bn_gc_reduction = nn.BatchNorm2d(gc_in_plane)
            self.nlc_convolution = nn.Conv1d((10 ** 2), (10 ** 2), (1, 3), padding=(0, 0))
            self.upsample = nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=False)
            self.num_dim = gc_in_plane - 2
            self.adaptiveavg2d = nn.AdaptiveAvgPool2d(10)

        else:
            self.num_dim = 0

        self.planes = planes
        self.inplanes = inplanes
        self.gc_group = gc_group
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes + self.num_dim)
        self.conv3 = nn.Conv2d(planes + self.num_dim, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.gc_group:
            gc_ = self.conv_gc_reduction(out)
            gc_ = self.bn_gc_reduction(gc_)
            gc_ = self.relu(gc_)
            gc_ = self.adaptiveavg2d(gc_)

            gc = torch.reshape(gc_, (gc_.shape[0], gc_.shape[1], 1, gc_.shape[2] * gc_.shape[3]))
            gc = gc.permute(0, 3, 2, 1)

            gc = self.nlc_convolution(gc)

            gc = gc.permute(0, 3, 2, 1)
            gcs = torch.reshape(gc, (gc_.shape[0], gc_.shape[1] - 2, gc_.shape[2], gc_.shape[3]))

            gcs = self.upsample(gcs)

        out = self.conv2(out)

        if self.gc_group:
            out = torch.cat((out, gcs), dim=1)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, dataset, depth, num_classes, bottleneck=False,
                 gc_group=[False, False, False, False]):

        super(ResNet, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        if self.dataset.startswith('cifar'):
            self.inplanes = 16
            print(bottleneck)
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock

            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 16, n, gc_group=gc_group[0], resolution=32)
            self.layer2 = self._make_layer(block, 32, n, stride=2, gc_group=gc_group[1], resolution=16)
            self.layer3 = self._make_layer(block, 64, n, stride=2, gc_group=gc_group[2], resolution=8)
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(64 * block.expansion, num_classes)

        elif dataset == 'imagenet':
            blocks = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
            layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3],
                      200: [3, 24, 36, 3]}
            assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0], gc_group=gc_group[0],
                                           resolution=56)
            self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2, gc_group=gc_group[1],
                                           resolution=28)
            self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2, gc_group=gc_group[2],
                                           resolution=14)
            self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2, gc_group=gc_group[3],
                                           resolution=7)
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, gc_group=False,
                    resolution=-1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            gc_group=gc_group,
                            resolution=resolution))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                gc_group=gc_group,
                                resolution=resolution))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        elif self.dataset == 'imagenet':
            x = self.conv1(x)

            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x