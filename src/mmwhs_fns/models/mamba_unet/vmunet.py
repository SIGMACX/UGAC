from .vmamba import VSSM
import torch
from torch import nn
# from .resvmamba import VSSM as Resvssm


class VMUNet(nn.Module):
    def __init__(self,
                 input_channels,
                 patch_size = 4,
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                ):
        super().__init__()

        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           patch_size = patch_size)
        
        # self.vmunet = Resvssm(in_chans=input_channels,
        #                    num_classes=num_classes,
        #                    depths=depths,
        #                    depths_decoder=depths_decoder,
        #                    drop_path_rate=drop_path_rate,
        #                    patch_size = patch_size)
    
    def forward(self, x):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        logits = self.vmunet(x)
        if self.num_classes == 1: return torch.sigmoid(logits)
        else: return logits, logits
