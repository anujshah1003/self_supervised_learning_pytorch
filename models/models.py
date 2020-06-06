# -*- coding: utf-8 -*-
import torchvision.models as models
import torch.nn as nn
def get_model(params,pretrained=False):
    if params.network=='resnet18':
        model = models.resnet18(pretrained=pretrained)
        if params.pretext=='rotation':
            params.num_classes=params.num_rot
        model.fc = nn.Linear(in_features=512,out_features=params.num_classes,bias=True)
        return model

def load_checkpoint(model,checkpoint_path,device):
    pass