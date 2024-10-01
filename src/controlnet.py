import torch.nn as nn
from unet.unet import UNet2DConditionModel

def make_zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class ControlNet(nn.Module):
    def __init__(self, frozen_model_config,
                 frozen_model_state_dict,
                 img_to_latent_downsample_factor
                 ):
        super().__init__()
        self.control_net_config = frozen_model_config
        self.control_net_config['disable_up_blocks'] = True
        self.control_net_config['return_controlnet_residuals'] = True
        if 'disable_up_blocks' in self.control_net_config['_use_default_values']:
            self.control_net_config['_use_default_values'].remove(
                'disable_up_blocks')
        if 'return_controlnet_residuals' in self.control_net_config['_use_default_values']:
            self.control_net_config['_use_default_values'].remove(
                'return_controlnet_residuals')

        self.controlnet_unet = UNet2DConditionModel.from_config(
            self.control_net_config)

        self.controlnet_unet.load_state_dict(frozen_model_state_dict)

        # 1 channel for canny
        hint_in_ch = 1
        base_hint_channel = 16
        curr_down_sample_factor = img_to_latent_downsample_factor
        hint_layers = [nn.Sequential(
                nn.Conv2d(hint_in_ch,
                          base_hint_channel,
                          kernel_size=3,
                          padding=(1, 1)),
                nn.SiLU())]
        while curr_down_sample_factor > 1:
            hint_layers.append(nn.Sequential(
                nn.Conv2d(base_hint_channel,
                          base_hint_channel*2,
                          kernel_size=3,
                          padding=(1, 1),
                          stride=2),
                nn.SiLU(),
                nn.Conv2d(base_hint_channel*2,
                          base_hint_channel*2,
                          kernel_size=3,
                          padding=(1, 1))
            ))
            base_hint_channel = base_hint_channel * 2
            curr_down_sample_factor = curr_down_sample_factor / 2
        hint_layers.append(nn.Sequential(
            nn.Conv2d(base_hint_channel,
                      self.controlnet_unet.down_channels[0],
                      kernel_size=3,
                      padding=(1, 1)),
            nn.SiLU(),
            make_zero_module(nn.Conv2d(self.controlnet_unet.down_channels[0],
                                       self.controlnet_unet.down_channels[0],
                                       kernel_size=1,
                                       padding=0))
        ))
        self.control_unet_hint_block = nn.Sequential(*hint_layers)

        # Zero Convolution Module for Downblocks(encoder Layers)
        self.control_unet_down_zero_convs = nn.ModuleList([
            make_zero_module(nn.Conv2d(self.controlnet_unet.down_channels[i],
                                       self.controlnet_unet.down_channels[i],
                                       kernel_size=1,
                                       padding=0))
            for i in range(len(self.controlnet_unet.down_channels)-1)
        ])

        # Zero Convolution Module for MidBlocks
        self.control_unet_mid_zero_convs = nn.ModuleList([
            make_zero_module(nn.Conv2d(self.controlnet_unet.mid_channels[i],
                                       self.controlnet_unet.mid_channels[i],
                                       kernel_size=1,
                                       padding=0))
            for i in range(1, len(self.controlnet_unet.mid_channels))
        ])
    
    def forward(self,latent, timestep, encoder_hidden_states, hint_img, added_cond_kwargs=None):
        control_unet_hint_out = self.control_unet_hint_block(hint_img)
        down_res, mid_res = self.controlnet_unet(latent, 
                                                 timestep, 
                                                 encoder_hidden_states, 
                                                 control_unet_hint_out, 
                                                 added_cond_kwargs=added_cond_kwargs)
        zero_conv_down_res = []
        zero_conv_mid_res = []
        for i in range(down_res):
            zero_conv_down_res.append(self.control_unet_down_zero_convs[i](down_res[i]))
        
        for i in range(mid_res):
            zero_conv_mid_res.append(self.control_unet_down_zero_convs[i](mid_res[i]))
        
        return zero_conv_down_res, zero_conv_mid_res

