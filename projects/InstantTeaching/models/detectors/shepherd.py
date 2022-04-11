import torch.nn as nn
import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, bbox_overlaps, multiclass_nms
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck


class ShepherdNet(nn.Module):
    def __init__(self, config=None):
        super(ShepherdNet, self).__init__()
        self.backbone = build_backbone(config.backbone)
        self.fc = nn.Linear(25088, 80)
    
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x[0], 1)
        x = self.fc(x)
        return x

