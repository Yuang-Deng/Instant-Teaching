import torch.nn as nn
import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, bbox_overlaps, multiclass_nms
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck


class ShepherdNet(nn.Module):
    def __init__(self, config=None):
        super(ShepherdNet, self).__init__()
        self.backbone = build_backbone(config.backbone)
        self.fc = nn.Linear(25088, 80)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        for module_list in [self.fc]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x[0], 1)
        x = self.fc(x)
        return x

