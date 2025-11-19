import time
import math
from functools import partial
from typing import Optional, Callable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# Import from the original files
from models.vmamba_Fusion_efficross import PatchEmbed2D, VSSLayer, PatchMerging2D, VSSLayer_up, PatchExpand2D, Final_PatchExpand2D
from models.cross import VSSBlock_Cross_new, VSSBlock_new

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except ImportError:
    print("Warning: mamba_ssm not installed. Please install it with: pip install mamba-ssm")

# In saeid_model.py, replace the SSIM section with:
has_ssim = False
ssim_loss_fn = None
print("SSIM temporarily disabled - using MSE-based similarity for multi-channel data")

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)
        self._initialized = False
        
    def _initialize_weights(self, device):
        """Initialize weights on the correct device"""
        if not self._initialized:
            self.weightx.data = self.weightx.data.to(device)
            self.weighty.data = self.weighty.data.to(device)
            self._initialized = True
        
    def forward(self, x):
        # Initialize weights on the correct device
        self._initialize_weights(x.device)
        
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

# class Fusionloss(nn.Module):
#     def __init__(self):
#         super(Fusionloss, self).__init__()
#         self.sobelconv = Sobelxy()

#     def forward(self, image_vis, image_ir, generate_img):
#         image_y = image_vis[:,:1,:,:]
#         x_in_max = torch.max(image_y, image_ir)
#         wb0 = 0.5
#         wb1 = 0.5

#         # Use appropriate SSIM based on image size
#         if has_ssim:
#             # For small patches (32x32), use regular SSIM
#             ssim_loss_temp1 = ssim_loss_fn(generate_img, image_y)
#             ssim_loss_temp2 = ssim_loss_fn(generate_img, image_ir)
#         else:
#             # Simple SSIM fallback using MSE (inverted)
#             ssim_loss_temp1 = 1.0 - F.mse_loss(generate_img, image_y)
#             ssim_loss_temp2 = 1.0 - F.mse_loss(generate_img, image_ir)
            
#         ssim_value = wb0 * (1 - ssim_loss_temp1) + wb1 * (1 - ssim_loss_temp2)
#         loss_in = F.mse_loss(x_in_max, generate_img)
#         y_grad = self.sobelconv(image_y)
#         ir_grad = self.sobelconv(image_ir)
#         generate_img_grad = self.sobelconv(generate_img)
#         x_grad_joint = torch.max(y_grad, ir_grad)
#         loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        
#         # Original weights from paper: ssim:10, intensity:10, gradient:1
#         loss_total = (10 * ssim_value) + (10 * loss_in) + (1 * loss_grad)
        
#         return loss_total, loss_in, ssim_value, loss_grad
class MultiChannelFusionLoss(nn.Module):
    def __init__(self):
        super(MultiChannelFusionLoss, self).__init__()
        self.sobelconv = Sobelxy()
        
        # Initialize SSIM for multi-channel
        try:
            from pytorch_msssim import SSIM
            self.ssim_loss_fn = SSIM(data_range=1.0, size_average=True)  # No channel specified - handles multi-channel
            self.has_ssim = True
        except ImportError:
            self.has_ssim = False
            self.ssim_loss_fn = None

    def forward(self, image_vis, image_ir, generate_img):
        """
        Multi-channel fusion loss
        """
        # For multi-channel data, we can either:
        # 1. Compute SSIM per channel and average
        # 2. Convert to single channel for SSIM computation
        
        # Option: Convert multi-channel to single channel using weighted average
        if image_vis.shape[1] > 1:
            # Use mean across channels for SSIM computation
            image_vis_single = image_vis.mean(dim=1, keepdim=True)
        else:
            image_vis_single = image_vis
            
        if image_ir.shape[1] > 1:
            image_ir_single = image_ir.mean(dim=1, keepdim=True)
        else:
            image_ir_single = image_ir
            
        if generate_img.shape[1] > 1:
            generate_img_single = generate_img.mean(dim=1, keepdim=True)
        else:
            generate_img_single = generate_img

        # Original fusion loss computation on single-channel representations
        x_in_max = torch.max(image_vis_single, image_ir_single)
        wb0 = 0.5
        wb1 = 0.5

        # Use appropriate SSIM
        if self.has_ssim:
            ssim_loss_temp1 = self.ssim_loss_fn(generate_img_single, image_vis_single)
            ssim_loss_temp2 = self.ssim_loss_fn(generate_img_single, image_ir_single)
        else:
            ssim_loss_temp1 = 1.0 - F.mse_loss(generate_img_single, image_vis_single)
            ssim_loss_temp2 = 1.0 - F.mse_loss(generate_img_single, image_ir_single)
            
        ssim_value = wb0 * (1 - ssim_loss_temp1) + wb1 * (1 - ssim_loss_temp2)
        loss_in = F.mse_loss(x_in_max, generate_img_single)
        
        # Gradient loss
        y_grad = self.sobelconv(image_vis_single)
        ir_grad = self.sobelconv(image_ir_single)
        generate_img_grad = self.sobelconv(generate_img_single)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        
        # Original weights from paper: ssim:10, intensity:10, gradient:1
        loss_total = (10 * ssim_value) + (10 * loss_in) + (1 * loss_grad)
        
        return loss_total, loss_in, ssim_value, loss_grad


class ChannelAwareFusionLoss(nn.Module):
    """
    Even better: Compute fusion loss per channel and average
    """
    def __init__(self):
        super(ChannelAwareFusionLoss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img):
        """
        Compute fusion loss across all channels
        
        Args:
            image_vis: HSI data [B, C_vis, H, W] 
            image_ir: LiDAR data [B, C_ir, H, W] 
            generate_img: Fused output [B, C_fused, H, W]
        """
        # Ensure we have the same number of channels for all inputs
        min_channels = min(image_vis.shape[1], image_ir.shape[1], generate_img.shape[1])
        
        # Use only the first min_channels to ensure compatibility
        image_vis = image_vis[:, :min_channels, :, :]
        image_ir = image_ir[:, :min_channels, :, :]
        generate_img = generate_img[:, :min_channels, :, :]
        
        # Compute losses per channel
        total_loss = 0
        intensity_loss = 0
        ssim_loss_val = 0
        gradient_loss = 0
        
        wb0 = 0.5
        wb1 = 0.5
        
        for c in range(min_channels):
            vis_channel = image_vis[:, c:c+1, :, :]
            ir_channel = image_ir[:, c:c+1, :, :]
            fused_channel = generate_img[:, c:c+1, :, :]
            
            # Intensity loss
            x_in_max = torch.max(vis_channel, ir_channel)
            loss_in = F.mse_loss(x_in_max, fused_channel)
            intensity_loss += loss_in
            
            # SSIM loss
            if has_ssim:
                ssim_temp1 = ssim_loss_fn(fused_channel, vis_channel)
                ssim_temp2 = ssim_loss_fn(fused_channel, ir_channel)
            else:
                ssim_temp1 = 1.0 - F.mse_loss(fused_channel, vis_channel)
                ssim_temp2 = 1.0 - F.mse_loss(fused_channel, ir_channel)
                
            ssim_val = wb0 * (1 - ssim_temp1) + wb1 * (1 - ssim_temp2)
            ssim_loss_val += ssim_val
            
            # Gradient loss
            vis_grad = self.sobelconv(vis_channel)
            ir_grad = self.sobelconv(ir_channel)
            fused_grad = self.sobelconv(fused_channel)
            joint_grad = torch.max(vis_grad, ir_grad)
            loss_grad = F.l1_loss(joint_grad, fused_grad)
            gradient_loss += loss_grad
            
            # Total loss for this channel
            channel_loss = (10 * ssim_val) + (10 * loss_in) + (1 * loss_grad)
            total_loss += channel_loss
        
        # Average across channels
        total_loss = total_loss / min_channels
        intensity_loss = intensity_loss / min_channels
        ssim_loss_val = ssim_loss_val / min_channels
        gradient_loss = gradient_loss / min_channels
        
        return total_loss, intensity_loss, ssim_loss_val, gradient_loss
    


class STEM_Block(nn.Module):
    """
    STEM block to convert input channels to target_channels (default: 64)
    Uses depthwise separable convolutions for efficient channel transformation
    """
    def __init__(self, in_channels, target_channels=64, kernel_size=3, stride=1, padding=1):
        super(STEM_Block, self).__init__()
        self.target_channels = target_channels
        
        # For HSI with many channels: use depthwise separable convolution
        if in_channels > target_channels:
            # Reduction path: many channels -> target_channels
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.GELU(),
                nn.Conv2d(in_channels, target_channels, 1, 1, 0),
                nn.BatchNorm2d(target_channels),
                nn.GELU()
            )
        elif in_channels < target_channels:
            # Expansion path: few channels -> target_channels
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels, target_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(target_channels),
                nn.GELU(),
                nn.Conv2d(target_channels, target_channels, kernel_size, 1, padding),
                nn.BatchNorm2d(target_channels),
                nn.GELU()
            )
        else:
            # Same number of channels, just apply convolution
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels, target_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(target_channels),
                nn.GELU()
            )
            
    def forward(self, x):
        return self.conv_layers(x)


class VSSM_Fusion_Classifier(nn.Module):
    def __init__(self, patch_size=4, in_chans1=1, in_chans2=1, num_classes=1000, 
                 depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], 
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, 
                 classifier_dropout=0.1, stem_channels=64, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.stem_channels = stem_channels

        # Store patch size for central pixel extraction
        self.patch_size = patch_size

        # STEM layers for channel adaptation
        self.stem1 = STEM_Block(in_chans1, stem_channels)  # For HSI (many bands -> 64)
        self.stem2 = STEM_Block(in_chans2, stem_channels)  # For LiDAR (1 band -> 64)

        # Dual encoder pathways - now with stem_channels as input
        self.patch_embed1 = PatchEmbed2D(patch_size=patch_size, in_chans=stem_channels, 
                                        embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)
        self.patch_embed2 = PatchEmbed2D(patch_size=patch_size, in_chans=stem_channels, 
                                        embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        # Absolute position embedding
        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed1.patches_resolution
            self.absolute_pos_embed1 = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            self.absolute_pos_embed2 = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed1, std=.02)
            trunc_normal_(self.absolute_pos_embed2, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        # Encoder layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # Decoder layers
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        # Cross-attention fusion blocks
        self.Cross_block = nn.ModuleList()
        for cross_layer in range(self.num_layers):
            clayer = VSSBlock_Cross_new(
                hidden_dim=dims[cross_layer],
                drop_path=drop_rate,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
            )
            self.Cross_block.append(clayer)

        # Final upsampling and convolution
        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, stem_channels, 1)  # Output stem_channels

        # Classifier head for central pixel classification
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(dims[-1]),
            nn.Dropout(classifier_dropout),
            nn.Linear(dims[-1], dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(dims[-1] // 2, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed1', 'absolute_pos_embed2'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features_1(self, x):
        """Forward pass for first modality (e.g., HSI)"""
        skip_list = []
        x = self.patch_embed1(x)
        if self.ape:
            x = x + self.absolute_pos_embed1
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list

    def forward_features_2(self, x):
        """Forward pass for second modality (e.g., LiDAR)"""
        skip_list = []
        x = self.patch_embed2(x)
        if self.ape:
            x = x + self.absolute_pos_embed2
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list

    def Fusion_network(self, skip_list1, skip_list2):
        """Fuse features from both modalities at multiple scales"""
        fused_skip_list = []
        for Cross_layer, skip1, skip2 in zip(self.Cross_block, skip_list1, skip_list2):
            fused_skip = Cross_layer(skip1, skip2)
            fused_skip_list.append(fused_skip)
        return fused_skip_list

    def forward_features_up(self, x, skip_list):
        """Decoder forward pass"""
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])
        return x

    def forward_final(self, x):
        """Final upsampling and convolution"""
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        x = self.final_conv(x)
        return x

    def extract_central_features(self, x, original_input_shape):
        """
        Extract features corresponding to central pixels of original patches
        
        Args:
            x: Feature map of shape [B, H, W, C] from last encoder layer
            original_input_shape: Original input shape [B, C, H, W]
            
        Returns:
            central_features: Features for central pixels [B, C]
        """
        B, _, orig_H, orig_W = original_input_shape
        B, H, W, C = x.shape
        
        # Calculate the center position in the feature map that corresponds to 
        # the center of the original input after all downsampling
        center_h, center_w = H // 2, W // 2
        
        # If we have multiple center positions, we can average them
        if H > 1 and W > 1:
            # Take 2x2 center region and average
            h_start = max(0, center_h - 1)
            w_start = max(0, center_w - 1)
            h_end = min(H, center_h + 1)
            w_end = min(W, center_w + 1)
            
            center_features = x[:, h_start:h_end, w_start:w_end, :].mean(dim=(1, 2))
        else:
            # Just take the single center point
            center_features = x[:, center_h, center_w, :]
        
        return center_features

    def forward(self, x1, x2):
        """
        Forward pass with feature extraction for classification
        
        Args:
            x1: First modality input (e.g., HSI) [B, C1, H, W] - many channels
            x2: Second modality input (e.g., LiDAR) [B, C2, H, W] - 1 channel
            
        Returns:
            dict: Contains all outputs including reconstruction and classification
        """
        # Store original inputs for residual connections (before STEM)
        x_1_orig, x_2_orig = x1, x2
        
        # Apply STEM layers to convert channels to stem_channels (64)
        x1_stem = self.stem1(x1)  # [B, C1, H, W] -> [B, 64, H, W]
        x2_stem = self.stem2(x2)  # [B, C2, H, W] -> [B, 64, H, W]
        
        original_shape = x1_stem.shape  # Use stem output shape for feature extraction

        # Encoder pathways
        x1_encoded, skip_list1 = self.forward_features_1(x1_stem)
        x2_encoded, skip_list2 = self.forward_features_2(x2_stem)

        # Fusion at encoder output - this is the last encoder layer fused features
        x_fused_encoder = x1_encoded + x2_encoded
        
        # Extract fused features from last encoder layer for classification
        fused_features_last_encoder = x_fused_encoder  # This is the last encoder output [B, H, W, C]
        
        # Extract features for central pixel classification
        central_features = self.extract_central_features(fused_features_last_encoder, original_shape)
        
        # Classification output
        classification_logits = self.classifier_head(central_features)

        # Fusion network for skip connections
        fused_skip_list = self.Fusion_network(skip_list1, skip_list2)

        # Decoder pathway
        x_decoded = self.forward_features_up(x_fused_encoder, fused_skip_list)
        
        # Final reconstruction output
        reconstruction_output = self.forward_final(x_decoded)
        
        # Add residual connections using stem outputs
        reconstruction_output = reconstruction_output + x1_stem + x2_stem

        return {
            'reconstruction': reconstruction_output,
            'classification': classification_logits,
            'fused_features_last_encoder': fused_features_last_encoder,
            'central_features': central_features,
            'stem_outputs': {
                'modality1': x1_stem,
                'modality2': x2_stem
            },
            'encoder_outputs': {
                'modality1': x1_encoded,
                'modality2': x2_encoded, 
                'fused': x_fused_encoder
            },
            'skip_lists': {
                'modality1': skip_list1,
                'modality2': skip_list2,
                'fused': fused_skip_list
            }
        }


# class CombinedFusionClassificationLoss(nn.Module):
#     def __init__(self, fusion_weight=1.0, classification_weight=1.0):
#         super().__init__()
#         self.fusion_weight = fusion_weight
#         self.classification_weight = classification_weight
#         self.fusion_loss = Fusionloss()
#         self.classification_loss = nn.CrossEntropyLoss()

#     def forward(self, outputs, x1, x2, classification_targets):
#         """
#         Compute combined loss with original fusion losses and classification loss
        
#         Args:
#             outputs: Dictionary from VSSM_Fusion_Classifier.forward()
#             x1: First modality input (HSI with many channels)
#             x2: Second modality input (LiDAR with 1 channel) 
#             classification_targets: Central pixel classification labels
#         """
#         # Compute original fusion losses
#         # Note: The fusion loss expects single-channel inputs, so we take the first channel
#         # from stem outputs for compatibility
#         fusion_total, fusion_intensity, fusion_ssim, fusion_grad = self.fusion_loss(
#             outputs['stem_outputs']['modality1'][:,:1,:,:],  # Take first channel from stem output
#             outputs['stem_outputs']['modality2'][:,:1,:,:],  # Take first channel from stem output
#             outputs['reconstruction'][:,:1,:,:]  # Take first channel from reconstruction
#         )
        
#         # Compute classification loss
#         class_loss = self.classification_loss(outputs['classification'], classification_targets)
        
#         # Combine losses
#         total_loss = (self.fusion_weight * fusion_total + 
#                      self.classification_weight * class_loss)
        
#         return {
#             'total_loss': total_loss,
#             'fusion_total': fusion_total,
#             'fusion_intensity': fusion_intensity,
#             'fusion_ssim': fusion_ssim,
#             'fusion_grad': fusion_grad,
#             'classification_loss': class_loss,
#             'loss_components': {
#                 'fusion_total': fusion_total.item(),
#                 'fusion_intensity': fusion_intensity.item(),
#                 'fusion_ssim': fusion_ssim.item(),
#                 'fusion_grad': fusion_grad.item(),
#                 'classification': class_loss.item()
#             }
#         }


class CombinedFusionClassificationLoss(nn.Module):
    def __init__(self, fusion_weight=1.0, classification_weight=1.0, use_channel_aware=True):
        super().__init__()
        self.fusion_weight = fusion_weight
        self.classification_weight = classification_weight
        
        # Choose between multi-channel fusion losses
        if use_channel_aware:
            self.fusion_loss = ChannelAwareFusionLoss()
        else:
            self.fusion_loss = MultiChannelFusionLoss()
            
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, x1, x2, classification_targets):
        """
        Compute combined loss with proper multi-channel fusion losses
        """
        # Use the full stem outputs (all channels) for fusion loss
        fusion_total, fusion_intensity, fusion_ssim, fusion_grad = self.fusion_loss(
            outputs['stem_outputs']['modality1'],  # All HSI channels
            outputs['stem_outputs']['modality2'],  # All LiDAR channels  
            outputs['reconstruction']              # All fused channels
        )
        
        # Compute classification loss
        class_loss = self.classification_loss(outputs['classification'], classification_targets)
        
        # Combine losses
        total_loss = (self.fusion_weight * fusion_total + 
                     self.classification_weight * class_loss)
        
        return {
            'total_loss': total_loss,
            'fusion_total': fusion_total,
            'fusion_intensity': fusion_intensity,
            'fusion_ssim': fusion_ssim,
            'fusion_grad': fusion_grad,
            'classification_loss': class_loss,
            'loss_components': {
                'fusion_total': fusion_total.item(),
                'fusion_intensity': fusion_intensity.item(),
                'fusion_ssim': fusion_ssim.item(),
                'fusion_grad': fusion_grad.item(),
                'classification': class_loss.item()
            }
        }

def main():
    """
    Main function to test the model with multi-channel HSI and single-channel LiDAR
    """
    print("Testing VSSM_Fusion_Classifier with STEM layers...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with different input channels
    num_classes = 10
    hsi_channels = 144  # Example: Houston2013 HSI has 144 bands
    lidar_channels = 1   # LiDAR has 1 band
    
    model = VSSM_Fusion_Classifier(
        patch_size=4,
        in_chans1=hsi_channels,   # HSI input channels
        in_chans2=lidar_channels, # LiDAR input channels
        num_classes=num_classes,
        depths=[2, 2, 9, 2],
        depths_decoder=[2, 9, 2, 2],
        dims=[96, 192, 384, 768],
        dims_decoder=[768, 384, 192, 96],
        d_state=16,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        classifier_dropout=0.2,
        stem_channels=64,  # Convert all inputs to 64 channels
        use_checkpoint=False
    ).to(device)
    
    # Print model summary
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy data with realistic channel dimensions
    batch_size = 2
    hsi_dummy = torch.randn(batch_size, hsi_channels, 32, 32).to(device)  # HSI: many channels
    lidar_dummy = torch.randn(batch_size, lidar_channels, 32, 32).to(device)  # LiDAR: 1 channel
    central_labels_dummy = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    print(f"\nInput shapes:")
    print(f"HSI data: {hsi_dummy.shape} ({hsi_channels} channels)")
    print(f"LiDAR data: {lidar_dummy.shape} ({lidar_channels} channels)")
    print(f"Central labels: {central_labels_dummy.shape}")
    
    # Forward pass
    print(f"\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(hsi_dummy, lidar_dummy)
    
    # Print output shapes
    print(f"\nOutput shapes:")
    print(f"Reconstruction: {outputs['reconstruction'].shape}")
    print(f"Classification logits: {outputs['classification'].shape}")
    print(f"Fused features last encoder: {outputs['fused_features_last_encoder'].shape}")
    print(f"Central features: {outputs['central_features'].shape}")
    print(f"STEM output HSI: {outputs['stem_outputs']['modality1'].shape}")
    print(f"STEM output LiDAR: {outputs['stem_outputs']['modality2'].shape}")
    
    # Test combined loss calculation
    print(f"\nTesting combined loss calculation...")
    criterion = CombinedFusionClassificationLoss(
        fusion_weight=1.0,
        classification_weight=1.0
    )
    
    losses = criterion(outputs, hsi_dummy, lidar_dummy, central_labels_dummy)
    
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Fusion total: {losses['fusion_total'].item():.4f}")
    print(f"Fusion intensity: {losses['fusion_intensity'].item():.4f}")
    print(f"Fusion SSIM: {losses['fusion_ssim'].item():.4f}")
    print(f"Fusion gradient: {losses['fusion_grad'].item():.4f}")
    print(f"Classification loss: {losses['classification_loss'].item():.4f}")
    
    # Test training step with combined losses
    print(f"\nTesting training step with combined losses...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Single training step
    optimizer.zero_grad()
    outputs = model(hsi_dummy, lidar_dummy)
    losses = criterion(outputs, hsi_dummy, lidar_dummy, central_labels_dummy)
    losses['total_loss'].backward()
    optimizer.step()
    
    print(f"Training step completed successfully!")
    print(f"Final losses:")
    print(f"  Total: {losses['total_loss'].item():.4f}")
    print(f"  Fusion total: {losses['fusion_total'].item():.4f}")
    print(f"  Fusion intensity: {losses['fusion_intensity'].item():.4f}")
    print(f"  Fusion SSIM: {losses['fusion_ssim'].item():.4f}")
    print(f"  Fusion gradient: {losses['fusion_grad'].item():.4f}")
    print(f"  Classification: {losses['classification_loss'].item():.4f}")
    
    print(f"\nModel test completed successfully!")


# Initialize model with STEM
def create_fusion_classifier(num_classes, hsi_channels, lidar_channels, patch_size=4, stem_channels=64):
    model = VSSM_Fusion_Classifier(
        patch_size=patch_size,
        in_chans1=hsi_channels,
        in_chans2=lidar_channels,
        num_classes=num_classes,
        depths=[2, 2, 9, 2],
        depths_decoder=[2, 9, 2, 2],
        dims=[96, 192, 384, 768],
        dims_decoder=[768, 384, 192, 96],
        d_state=16,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        classifier_dropout=0.2,
        stem_channels=stem_channels
    )
    return model


if __name__ == "__main__":
    main()