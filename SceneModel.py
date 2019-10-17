from Place365 import wideresnet
from netVLADbase import netVLADbaseResNet

import torch
import torch.nn as nn
import numpy as np
from os.path import isfile

features_blobs = []
netVLADlayer_input = []  # netVLAD分支的输入值

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def hook_layer(module, input, output):
    netVLADlayer_input.append(output)

def getPretrainedParams(model_file):
    # load object saved with torch.save() from a file
    # with funtion specifiying how to remap storage locations in the parameter list
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) #gpu->cpu, why?!
    # get rid of ‘module.’
    state_dict = {str.replace(k, 'module.', ''): v for k,v in checkpoint['state_dict'].items()}
    return state_dict

# 构造场景识别/场景属性的resnet模型，并加载参数；同时增加netvlad分支所用的hook
def loadSceneRecognitionModel(trainedNetVLADLayers):
    # this model has a last conv feature map as 14x14
    model_file = 'Place365/wideresnet18_places365.pth.tar'
    model = wideresnet.resnet18(num_classes=365)
    state_dict = getPretrainedParams(model_file)
    model.load_state_dict(state_dict)
    model.eval()

    # hook the feature extractor
    features_names = ['layer4', 'avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)

    # hook the layer for DataSet
    features_candidates = ['layer4', 'layer3', 'layer2', 'layer1', 'relu1']
    model._modules.get(features_candidates[trainedNetVLADLayers]).register_forward_hook(hook_layer)

    return model

# 导入训练过的resnet-netvlad模型
def loadPlaceRecognitionEncoder(netVLADtrainNum):
    model = netVLADbaseResNet(wideresnet.BasicBlock, [2, 2, 2, 2], netVLADtrain=netVLADtrainNum)
    model_file = 'Place365/wideresnet18_places365.pth.tar'
    state_dict = getPretrainedParams(model_file)
    # 只保留 ‘layer’ 的部分
    for key in list(state_dict.keys()):
        if not 'layer' in key:
            del state_dict[key]
            continue
    model.load_state_dict(state_dict)
    layers = list(model.children())[-1*model.netVLADtrain:]
    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module('encoder', encoder)

    return model

# 读入netVLAD分支的预先训练结果
def loadNetVLADParams(resume_ckpt, netVLADtrainNum, model):
    if isfile(resume_ckpt):
        print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in checkpoint['state_dict'].items()}
        # 只保留 ‘layer’ 的部分，且更新编号
        for key in list(state_dict.keys()):
            if ('encoder.0' in key) or ('encoder.1' in key):
                del state_dict[key]
                continue
        if netVLADtrainNum == 4:
            state_dict = {str.replace(k,'encoder.3','encoder.0'): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, 'encoder.4', 'encoder.1'): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, 'encoder.5', 'encoder.2'): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, 'encoder.6', 'encoder.3'): v for k, v in state_dict.items()}
        if netVLADtrainNum <= 3:
            for key in list(state_dict.keys()):
                if 'encoder.3' in key:
                    del state_dict[key]
                    continue
            state_dict = {str.replace(k, 'encoder.4', 'encoder.0'): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, 'encoder.5', 'encoder.1'): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, 'encoder.6', 'encoder.2'): v for k, v in state_dict.items()}
        if netVLADtrainNum <= 2:
            for key in list(state_dict.keys()):
                if 'encoder.0' in key:
                    del state_dict[key]
                    continue
            state_dict = {str.replace(k, 'encoder.1', 'encoder.0'): v for k, v in state_dict.items()}
            state_dict = {str.replace(k, 'encoder.2', 'encoder.1'): v for k, v in state_dict.items()}
        if netVLADtrainNum <= 1:
            for key in list(state_dict.keys()):
                if 'encoder.0' in key:
                    del state_dict[key]
                    continue
            state_dict = {str.replace(k, 'encoder.1', 'encoder.0'): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_ckpt, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_ckpt))
    model.eval()

    return model
