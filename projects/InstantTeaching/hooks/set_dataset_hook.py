import copy
import warnings

from mmcv.cnn import VGG
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class SetDataSetHook(Hook):    
    def before_epoch(self, runner):
        if hasattr(runner.model.module, 'set_dataset'):
            runner.model.module.set_dataset(runner.data_loader.dataset)
        
