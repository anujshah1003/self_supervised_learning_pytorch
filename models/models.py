# -*- coding: utf-8 -*-
import torchvision.models as models
import torch.nn as nn
def get_model(params,pretrained=False):
    if params.network=='resnet18':
        model = models.resnet18(pretrained=pretrained)
        if params.pretext=='rotation':
            params.num_classes=params.num_rot
        model.fc = nn.Linear(in_features=model.fc.in_features,out_features=params.num_classes,bias=True)
        return model

def load_checkpoint(model,checkpoint_path,device):
    pass

class LogisticRegression(nn.Module):
    
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)