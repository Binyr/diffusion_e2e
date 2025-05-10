from pathlib import Path
from diffusers import UNetSpatioTemporalConditionModel
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionOutput
from diffusers.configuration_utils import FrozenDict, register_to_config, ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version
import torch
import json
from typing import Dict, Optional, Tuple, Union

class DINOv2_Encoder(torch.nn.Module):
    IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_name = 'dinov2_vitl14',
        freeze = True,
        antialias=True,
        device="cuda",
        size = 448,
    ):
        super(DINOv2_Encoder, self).__init__()
        
        # self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        torch.hub.set_dir("opensource_models")
        self.model = torch.hub.load('submodule/dinov2-main', model_name, pretrained=True, source="local")
        
        self.model.eval().to(device)
        self.device = device
        self.antialias = antialias
        self.dtype = torch.float32

        self.mean = torch.Tensor(self.IMAGENET_DEFAULT_MEAN)
        self.std = torch.Tensor(self.IMAGENET_DEFAULT_STD)
        self.size = size
        if freeze:
            self.freeze()


    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encoder(self, x, size=None):
        '''
        x: [b h w c], range from (-1, 1), rbg
        '''

        x = self.preprocess(x, size).to(self.device, self.dtype)

        b, c, h, w = x.shape
        patch_h, patch_w = h // 14, w // 14

        embeddings = self.model.forward_features(x)['x_norm_patchtokens']
        embeddings = rearrange(embeddings, 'b (h w) c -> b h w c', h = patch_h, w = patch_w)

        return  rearrange(embeddings, 'b h w c -> b c h w')

    def preprocess(self, x, size=None):
        ''' x
        '''
        # normalize to [0,1],
        if size is None:
            size = (self.size, self.size)
        x = torch.nn.functional.interpolate(
            x,
            size=size,
            mode='bicubic',
            align_corners=True,
            antialias=self.antialias,
        )

        x = (x + 1.0) / 2.0
        # renormalize according to dino
        mean = self.mean.view(1, 3, 1, 1).to(x.device)
        std = self.std.view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std

        return x
    
    def to(self, device, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.model.to(device, dtype)
            self.mean.to(device, dtype)
            self.std.to(device, dtype)
        else:
            self.model.to(device)
            self.mean.to(device)
            self.std.to(device)
        return self

    def __call__(self, x, **kwargs):
        return self.encoder(x, **kwargs)

class TrainPipeline(ModelMixin, ConfigMixin): # ConfigMixin, ModelMixin, torch.nn.Module
    # @register_to_config
    # def __init__(self, pretrained_name_or_path):
    def __init__(self, unet, image_controlnet=None, dino_controlnet=None,
        dino_distiller=None,
        unet_not_input_image=False,
        timestep_dino_controlnet_mode=None,
    ):
        super().__init__()

        # unet = IntrinsicUNetSpatioTemporalConditionModel.from_pretrained(
        #     pretrained_name_or_path,
        #     subfolder="unet",
        #     low_cpu_mem_usage=True,
        #     variant="fp16",
        # )
        self.unet = unet
        # import pdb
        # pdb.set_trace()
        # self.register_module(
        #     "image_controlnet", image_controlnet,
        # )
        self.image_controlnet = image_controlnet
        
        self.dino_controlnet = dino_controlnet

        self.dino_distiller = dino_distiller

        self.unet_not_input_image = unet_not_input_image
        self.timestep_dino_controlnet_mode = timestep_dino_controlnet_mode
        # to make this pipeline looks like unet
        # config = json.loads(unet.to_json_string())
        # self.register_to_config(**config)
    
    def save_pretrained(self, save_directory, **kwargs):
        unet_dir = str(Path(save_directory) / "unet")
        self.unet.save_pretrained(unet_dir, **kwargs)
        if self.image_controlnet is not None:
            controlnet_dir = str(Path(save_directory) / "image_controlnet")
            self.image_controlnet.save_pretrained(controlnet_dir)

        if self.dino_controlnet is not None:
            dino_controlnet_dir = str(Path(save_directory) / "dino_controlnet")
            self.dino_controlnet.save_pretrained(dino_controlnet_dir)
    
    def enable_xformers_memory_efficient_attention(self, ):
        # import pdb
        # pdb.set_trace()
        self.unet.enable_xformers_memory_efficient_attention()
        if self.image_controlnet is not None:
            self.image_controlnet.enable_xformers_memory_efficient_attention()
        if self.dino_controlnet is not None:
            self.dino_controlnet.enable_xformers_memory_efficient_attention()
    
    def enable_gradient_checkpointing(self, ):
        self.unet.enable_gradient_checkpointing()
        if self.image_controlnet is not None:
            self.image_controlnet.enable_gradient_checkpointing()
        if self.dino_controlnet is not None:
            self.dino_controlnet.enable_xformers_memory_efficient_attention()
    
    def forward(
        self,
        unet_input, timesteps, encoder_hidden_states,
        dino_feat_distills: torch.FloatTensor = None,
        return_dict: bool = False,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
       
        ret = self.unet(
            unet_input,
            timesteps,
            encoder_hidden_states,
            return_dict,
            )
        if not return_dict:
            ret, down_block_out_sampels, mid_up_block_out_samples = ret
            dino_distill_loss = None
            
            if self.dino_distiller:
                dino_distill_loss = self.dino_distiller(
                    dino_feat_distills, down_block_out_sampels + mid_up_block_out_samples
                )
        
            return ret, dino_distill_loss
        else:
            return ret