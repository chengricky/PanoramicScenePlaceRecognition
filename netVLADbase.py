# This file defines the CNN combined netVLAD module and scene catagory utility

# loading resnet18 of trained with netVLAD as the basenet
import torch.nn as nn
import math


class netVLADbaseResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, netVLADtrain=2):
        super(netVLADbaseResNet, self).__init__()
        self.inplanes = 64 ## 针对分支重新设定变量数值
        self.netVLADtrain = netVLADtrain
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.netVLADtrain >= 4:
            x = self.layer1(x)
        if self.netVLADtrain >= 3:
            x = self.layer2(x)
        if self.netVLADtrain >= 2:
            x = self.layer3(x)
        if self.netVLADtrain >= 1:
            x = self.layer4(x)
        return x




