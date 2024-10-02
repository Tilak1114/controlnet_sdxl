import torch.nn as nn
from unet.unet import UNet2DConditionModel
import torch 

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
        self.control_net_config['is_controlnet'] = True
        if 'disable_up_blocks' in self.control_net_config['_use_default_values']:
            self.control_net_config['_use_default_values'].remove(
                'disable_up_blocks')
        if 'is_controlnet' in self.control_net_config['_use_default_values']:
            self.control_net_config['_use_default_values'].remove(
                'is_controlnet')

        self.controlnet_unet = UNet2DConditionModel.from_config(
            self.control_net_config)

        self.controlnet_unet.load_state_dict(frozen_model_state_dict, strict=False)

        self.check_mapping(frozen_model_state_dict)
 
        hint_in_ch = 3
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
        
        hint_block_out_ch = self.get_channels(self.controlnet_unet.down_blocks[0])

        hint_layers.append(nn.Sequential(
            nn.Conv2d(base_hint_channel,
                      hint_block_out_ch,
                      kernel_size=3,
                      padding=(1, 1)),
            nn.SiLU(),
            make_zero_module(nn.Conv2d(hint_block_out_ch,
                                       hint_block_out_ch,
                                       kernel_size=1,
                                       padding=0))
        ))
        self.control_unet_hint_block = nn.Sequential(*hint_layers)

        # Zero Convolution Module for Downblocks(encoder Layers)
        down_zero_modules = [make_zero_module(nn.Conv2d(
                                        self.get_channels(self.controlnet_unet.down_blocks[0], ch_type='out'),
                                        self.get_channels(self.controlnet_unet.down_blocks[0], ch_type='out'),
                                        kernel_size=1,
                                        padding=0))]
        for i in range(len(self.controlnet_unet.down_blocks)):
            zero_ch = self.get_channels(self.controlnet_unet.down_blocks[i], ch_type='out')
            num_zero_modules = 0
            if self.controlnet_unet.down_blocks[i].resnets:
                num_zero_modules += len(self.controlnet_unet.down_blocks[i].resnets)
            if self.controlnet_unet.down_blocks[i].downsamplers:
                num_zero_modules += len(self.controlnet_unet.down_blocks[i].downsamplers)

            for _ in range(num_zero_modules):
                down_zero_modules.append(
                    make_zero_module(nn.Conv2d(zero_ch,
                                       zero_ch,
                                       kernel_size=1,
                                       padding=0))
                )
        self.control_unet_down_zero_convs = nn.ModuleList(down_zero_modules)

        # Zero Convolution Module for MidBlocks        
        zero_ch = self.get_channels(self.controlnet_unet.mid_block, ch_type='out')
        self.control_unet_mid_zero_convs = nn.ModuleList([
            make_zero_module(nn.Conv2d(zero_ch,
                                    zero_ch,
                                    kernel_size=1,
                                    padding=0))
        ])
    
    def get_channels(self, module, instance_type='conv', ch_type='in'):
        if instance_type == 'conv':
            instance = nn.Conv2d
            for layer in module.modules():
                if isinstance(layer, instance) and ch_type == 'in':
                    return layer.in_channels
                elif isinstance(layer, instance) and ch_type == 'out':
                    return layer.out_channels
        else:
            instance = nn.Linear
            for layer in module.modules():
                if isinstance(layer, instance) and ch_type == 'in':
                    return layer.in_features
                elif isinstance(layer, instance) and ch_type == 'out':
                    return layer.out_features

        return -1
    
    def check_mapping(self, frozen_model_state_dict):
        mismatched_keys = []
        missing_keys = []
        for key in frozen_model_state_dict.keys():
            if key in self.controlnet_unet.state_dict():
                if not torch.equal(self.controlnet_unet.state_dict()[key].to('cpu'), frozen_model_state_dict[key]):
                    mismatched_keys.append(key)
            else:
                missing_keys.append(key)
        print(f"Mismatched keys: {mismatched_keys}")
    
    def forward(self,latent, timestep, encoder_hidden_states, hint_img, added_cond_kwargs=None):
        control_unet_hint_out = self.control_unet_hint_block(hint_img)
        down_res, mid_res = self.controlnet_unet(latent, 
                                                 timestep, 
                                                 encoder_hidden_states, 
                                                 control_unet_hint_out, 
                                                 added_cond_kwargs=added_cond_kwargs)
        zero_conv_down_res = []
        zero_conv_mid_res = []
        for i in range(len(down_res)):
            zero_conv_down_res.append(self.control_unet_down_zero_convs[i](down_res[i]))
        
        zero_conv_mid_res = self.control_unet_mid_zero_convs[0](mid_res[0])
        
        return zero_conv_down_res, zero_conv_mid_res

