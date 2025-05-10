import torch
import torch.nn as nn

from diffusers.models.attention_processor import Attention

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

class TemperalAttention(nn.Module):
    def __init__(self, 
            dim,
            num_attention_heads,
            attention_head_dim,
        ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
        )
        self.attn1.set_use_memory_efficient_attention_xformers(True)
    
    def forward(self, dino_feat):
        N, T, C, H, W = dino_feat.shape
        init_mask = torch.eye(T, device=dino_feat.device, dtype=dino_feat.dtype).unsqueeze(0) * 5
        dino_feat = dino_feat.reshape(N, T, C, H*W)
        dino_feat = dino_feat.permute(0, 3, 1, 2)
        dino_feat = dino_feat.reshape(N*H*W, T, C)
        dino_feat = self.norm1(dino_feat)
        dino_feat = self.attn1(dino_feat, encoder_hidden_states=None, attention_mask=init_mask)
        dino_feat = dino_feat.reshape(N, H*W, T, C)
        dino_feat = dino_feat.permute(0, 2, 3, 1)
        dino_feat = dino_feat.reshape(N, T, C, H, W)
        return dino_feat

class DINODistiller(nn.Module):
    UNET_FEAT_NAMES=["down_0", "down_1", "down_2", "down_3", "mid", "up_0", "up_1", "up_2", "up_3"]
    def __init__(self,
            unet, 
            unet_feat_names=["mid"],
            unet_layer_dim=2048,
            dino_dim=1024,
            dino_layer_type=None, # can be "temperal_attention",
            loss_type="kl",
        ):
        super().__init__()
        for name in unet_feat_names:
            assert name in self.UNET_FEAT_NAMES
        self.unet_feat_names = unet_feat_names
        self.dino_layer_type = dino_layer_type
        self.dino_layers = nn.ModuleList()
        
        # num_attention_heads = unet.config.num_attention_heads[-1]
        attention_head_dim = 64
        num_attention_heads = dino_dim // attention_head_dim
        for i in range(len(unet_feat_names)):
            if dino_layer_type == "temperal_attention":
                dino_layer = TemperalAttention(
                    dino_dim,
                    num_attention_heads,
                    attention_head_dim
                )
            else:
                dino_layer = nn.Identity()
            self.dino_layers.append(dino_layer)

        self.projectors = nn.ModuleList()
        for feat_name in unet_feat_names:
            if feat_name == "mid":
                feat_dim = unet.config.block_out_channels[-1]
            elif feat_name.startswith("down"):
                block_idx = int(feat_name.split("_")[1])
                feat_dim = unet.config.block_out_channels[block_idx]
            elif feat_name.startswith("up"):
                block_idx = int(feat_name.split("_")[1])
                feat_dim = unet.config.block_out_channels[::-1][block_idx]
            else:
                feat_dim, num_attention_heads, attention_head_dim = None, None, None
            self.projectors.append(build_mlp(feat_dim, unet_layer_dim, dino_dim))
    
    def forward(self, dino_feats, unet_feats):
        if not isinstance(dino_feats, list):
            dino_feats = [dino_feats]
        
        # dino_feat = dino_feat.transposed(N*T, H*W, dC)
        proj_loss = 0.
        for i, feat_name in enumerate(self.unet_feat_names):
            dino_feat = self.dino_layers[i](dino_feats[i])
            N, T, dC, dH, dW = dino_feat.shape
            dino_feat = dino_feat.reshape(N*T, dC, dH, dW)


            feat_dix = self.UNET_FEAT_NAMES.index(feat_name)
            unet_feat = unet_feats[feat_dix]
            
            NT, C, H, W = unet_feat.shape

            unet_feat = unet_feat.reshape(NT, C, H*W)
            unet_feat = unet_feat.permute(0, 2, 1)
            unet_feat_mapped = self.projectors[i](unet_feat)
            
            unet_feat_mapped = unet_feat_mapped.permute(0, 2, 1)
            unet_feat_mapped = unet_feat_mapped.reshape(NT, -1, H, W)
            unet_feat_mapped = torch.nn.functional.interpolate(unet_feat_mapped, (dH, dW), mode="bicubic", align_corners=True, antialias=True)

            z_tilde_j = torch.nn.functional.normalize(unet_feat_mapped, dim=1) 
            z_j = torch.nn.functional.normalize(dino_feat, dim=1)
            proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=1))
        proj_loss = proj_loss.mean() / len(self.unet_feat_names)
        return proj_loss
    
    @staticmethod
    def parse_image_size_for_dino_net(unet_feat_names, unet_input_size):
        W, H = unet_input_size
        W = W // 8 // 8 * 14
        H = H // 8 // 8 * 14
        if not isinstance(unet_feat_names, list):
            unet_feat_names = [unet_feat_names]
        
        ret_sizes = []
        for unet_feat_name in unet_feat_names:
            if unet_feat_name in ["down_3", "mid"]:
                ret_sizes.append((W, H))
            elif unet_feat_name in ["down_2", "up_0"]:
                ret_sizes.append((W * 2, H * 2))
            elif unet_feat_name in ["down_1", "up_1"]:
                ret_sizes.append((W * 4, H * 4))
            elif unet_feat_name in ["down_0", "up_2", "up_3"]:
                ret_sizes.append((W * 8, H * 8))
        return ret_sizes

