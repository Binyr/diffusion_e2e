from diffusers import UNetSpatioTemporalConditionModel
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionOutput
from diffusers.configuration_utils import FrozenDict
from diffusers.utils import is_torch_version
import torch
from typing import Dict, Optional, Tuple, Union

def create_custom_forward(module, return_dict=None):
    def custom_forward(*inputs):
        if return_dict is not None:
            return module(*inputs, return_dict=return_dict)
        else:
            return module(*inputs)

    return custom_forward
CKPT_KWARGS = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}


class IntrinsicUNetSpatioTemporalConditionModel(UNetSpatioTemporalConditionModel):
    def post_init(self, new_out_channels, new_cond_channels):
         # custom_gradient_checkpointing
        self.custom_gradient_checkpointing = False

        # diffuse, metallic, roughness # depth, normal
        # new_out_channels = 12
        new_in_channels = new_out_channels + new_cond_channels

        old_conv_out = self.conv_out # (4, 320, 3, 3)
        old_out_channels = old_conv_out.weight.shape[0]
        old_conv_in = self.conv_in # (320, 8, 3, 3)
        old_in_channels = old_conv_in.weight.shape[1]
        if new_out_channels == old_out_channels and new_in_channels == old_in_channels:
            print(f"do nothing")
            return
        
        # 1. out layer
        if not (new_out_channels == old_out_channels):
            # 1.1. new layers
            conv_out = torch.nn.Conv2d(
                old_conv_out.weight.shape[1], # 320
                new_out_channels, 
                old_conv_out.kernel_size,
                old_conv_out.stride,
                old_conv_out.padding,
                device=old_conv_out.weight.device,
                dtype=old_conv_out.weight.dtype
            )
            conv_out.requires_grad_(False)

            # 1.2. init layer
            with torch.no_grad():
                for i in range(new_out_channels):
                    # init weight
                    conv_out.weight[i, ...] = old_conv_out.weight[i % old_out_channels, ...]
                    # init bias
                    conv_out.bias[i, ...] = old_conv_out.bias[i % old_out_channels, ...]
        
            conv_out.requires_grad_(old_conv_out.weight.requires_grad)
            self.conv_out = conv_out

        # import pdb
        # pdb.set_trace()
        # 2. input layer
        if not (new_in_channels == old_in_channels):
            conv_in = torch.nn.Conv2d(
                new_in_channels,
                old_conv_in.weight.shape[0], # 320, 
                old_conv_out.kernel_size,
                old_conv_out.stride,
                old_conv_out.padding,
                device=old_conv_out.weight.device,
                dtype=old_conv_out.weight.dtype
            )
            conv_in.requires_grad_(False)
            # new_out_channels means num_dimension_of_latent
            num_repeat = new_out_channels // old_out_channels
            with torch.no_grad():
                rgb_latent_channels = 4
                new_tgt_latent_channels = new_out_channels
                old_tgt_latent_channels = old_out_channels
                addition_latent_channels = new_in_channels - new_tgt_latent_channels - rgb_latent_channels
                # init weight of latents
                for i in range(new_tgt_latent_channels):
                    conv_in.weight[:, i, ...] = old_conv_in.weight[:, i % old_tgt_latent_channels, ...] / num_repeat
                    
                # init weight of rgb_latents
                for i in range(rgb_latent_channels): 
                    conv_in.weight[:, new_tgt_latent_channels + i, ...] = old_conv_in.weight[:, old_tgt_latent_channels + i, ...]
                
                # init weight of addition_latents
                for i in range(addition_latent_channels): 
                    conv_in.weight[:, new_tgt_latent_channels + rgb_latent_channels + i, ...] = 0.

                conv_in.bias[...] = old_conv_in.bias[...]
            conv_in.requires_grad_(old_conv_in.weight.requires_grad)
            self.conv_in = conv_in

        # 3. config
        # config needs to be modified, otherwise you can't reload from checkpoint. As old config is saved.
        old_config = self._internal_dict
        # _internal_dict is a FrozenDict, which can't be modified, recreate instead
        new_config = {k: v for k, v in old_config.items()}
        if "out_channels" in new_config:
            new_config["out_channels"] = new_out_channels
        if "in_channels" in new_config:
            new_config["in_channels"] = new_in_channels
        self._internal_dict = FrozenDict(new_config)
    
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.FloatTensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead of a plain
                tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
        if not hasattr(self, "custom_gradient_checkpointing"):
            self.custom_gradient_checkpointing = False
        # import pdb
        # pdb.set_trace()
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps) # (B, C)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb) # (B, C)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        # here, our encoder_hidden_states is [batch * frames, 1, channels]
        # print("#" * 100, encoder_hidden_states.shape)
        # import pdb
        # pdb.set_trace()
        if not sample.shape[0] == encoder_hidden_states.shape[0]:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # 2. pre-process
        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)
    
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if self.custom_gradient_checkpointing:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    sample, res_samples = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(downsample_block),
                        sample,
                        emb,
                        encoder_hidden_states,
                        image_only_indicator,
                        **CKPT_KWARGS,
                    )
                else:
                    sample, res_samples = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(downsample_block),
                        sample,
                        emb,
                        image_only_indicator,
                        **CKPT_KWARGS,
                    )
            else:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        image_only_indicator=image_only_indicator,
                    )
                else:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        image_only_indicator=image_only_indicator,
                    )

            down_block_res_samples += res_samples

        # 4. mid
        if self.custom_gradient_checkpointing:
            sample = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block),
                sample,
                emb,
                encoder_hidden_states,
                image_only_indicator,
                **CKPT_KWARGS,
        )
        else:
            sample = self.mid_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
            )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                if self.custom_gradient_checkpointing:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(upsample_block),
                        sample,
                        res_samples,
                        emb,
                        encoder_hidden_states,
                        image_only_indicator,
                        **CKPT_KWARGS
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        image_only_indicator=image_only_indicator,
                    )
            else:
                if self.custom_gradient_checkpointing:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(upsample_block),
                        sample,
                        res_samples,
                        emb,
                        image_only_indicator,
                        **CKPT_KWARGS
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        image_only_indicator=image_only_indicator,
                    )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        if self.custom_gradient_checkpointing:
            sample = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.conv_out),
                    sample,
                    **CKPT_KWARGS
                )
        else:
            sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample,)

        return UNetSpatioTemporalConditionOutput(sample=sample)