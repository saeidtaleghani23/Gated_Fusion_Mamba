# -*- coding:utf-8 -*-
# @Time       :2022/9/8 上午11:07
# @AUTHOR     :DingKexin
# @FileName   :CALC.py
"""
Coupled Adversarial Learning based Classification (CALC) Model

This module implements the CALC framework for multimodal remote sensing image
classification, combining hyperspectral (HSI) and LiDAR data through coupled
adversarial learning and multi-level feature fusion.

Architecture Overview:
----------------------
The CALC model consists of two main sub-networks:
1. CAFL (Coupled Adversarial Feature Learning) - Unsupervised feature fusion
2. MFFC (Multi-level Feature Fusion Classification) - Supervised classification

Key Components:
---------------
- Encoder: Extracts multi-level features from HSI and LiDAR with weight sharing
- Decoder: Reconstructs modality-specific images for adversarial learning  
- Classifier: Performs classification using fused multi-level features
- Discriminators: Preserve modality-specific details through adversarial training
- Spatial Attention: Enhances focus on discriminative spatial regions

Input Specifications:
---------------------
- HSI Patch: [batch_size, spectral_bands, 16, 16] (e.g., [64, 144, 16, 16])
- LiDAR Patch: [batch_size, lidar_bands, 16, 16] (e.g., [64, 1, 16, 16])

Feature Flow for 16×16×channels input:
---------------------------------------
Encoder Processing:
  Block 1: [B, C_hsi,16,16]→[B,32,8,8], [B,C_lidar,16,16]→[B,32,8,8] → Fusion: [B,32,8,8]
  Block 2: [B,32,8,8]→[B,64,4,4] (Weight-shared) → Fusion: [B,64,4,4]
  Block 3: [B,64,4,4]→[B,128,2,2] (Weight-shared) → Fusion: [B,128,2,2]

Decoder Processing (from [B,128,2,2]):
  Block 1: Upsample→[B,128,4,4]→Conv→[B,64,4,4]
  Block 2: Upsample→[B,64,8,8]→Conv→[B,32,8,8] 
  Block 3: 
    HSI: Upsample→[B,32,16,16]→Conv→[B,C_hsi,16,16]
    LiDAR: Upsample→[B,32,16,16]→Conv→[B,C_lidar,16,16]

Classifier Processing:
  High-level ([B,128,2,2]): Conv→[B,64,2,2]→Conv+Pool→[B,32,1,1]→Conv→[B,Classes,1,1]→[B,Classes]
  Mid-level ([B,64,4,4]): Conv→[B,32,4,4]→Conv+Pool→[B,16,1,1]→Conv→[B,Classes,1,1]→[B,Classes]
  Low-level ([B,32,8,8]): Conv→[B,16,8,8]→Conv+Pool→[B,16,1,1]→Conv→[B,Classes,1,1]→[B,Classes]

Final Output: Weighted fusion of three classification probabilities

Reference:
----------
Lu, T., Ding, K., Fu, W., Li, S., & Guo, A. (2023). 
Coupled adversarial learning for fusion classification of hyperspectral and LiDAR data. 
Information Fusion, 93, 118-131.
"""

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    
    Enhances discriminative spatial regions by computing attention weights
    through channel-wise average and max pooling, followed by convolution.
    
    Input: [batch, channels, height, width]
    Output: [batch, 1, height, width] (spatial attention weights)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise average and max pooling
        avgout = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
        maxout, _ = torch.max(x, dim=1, keepdim=True)  # [B,1,H,W]
        # Concatenate and compute spatial attention
        x = torch.cat([avgout, maxout], dim=1)  # [B,2,H,W]
        x = self.conv(x)  # [B,1,H,W]
        return self.sigmoid(x)  # [B,1,H,W]


class Encoder(nn.Module):
    """
    Dual-branch Encoder with Weight Sharing
    
    Extracts multi-level features from HSI and LiDAR data with weight sharing
    in higher layers to capture similar semantic information.
    
    Inputs:
        x1: HSI patches [batch, l1, 16, 16]
        x2: LiDAR patches [batch, l2, 16, 16]
    
    Outputs (three fused feature levels):
        x1_add: Low-level features [batch, 32, 8, 8]
        x2_add: Mid-level features [batch, 64, 4, 4] 
        x3_add: High-level features [batch, 128, 2, 2]
    """
    def __init__(self, l1, l2):
        """
        Args:
            l1: Number of HSI spectral bands
            l2: Number of LiDAR bands
        """
        super(Encoder, self).__init__()
        # First block: Separate processing for each modality
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=l1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 16×16 → 8×8
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=l2, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 16×16 → 8×8
        )
        
        # Second and third blocks: Weight-shared processing
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8×8 → 4×4
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4×4 → 2×2
        )
        
        # Learnable fusion coefficients
        self.xishu1 = torch.nn.Parameter(torch.Tensor([0.5]))  # HSI weight
        self.xishu2 = torch.nn.Parameter(torch.Tensor([0.5]))  # LiDAR weight

    def forward(self, x1, x2):
        # Block 1: Separate processing + fusion
        x1_1 = self.conv1_1(x1)  # [B,32,8,8]
        x1_2 = self.conv1_2(x2)  # [B,32,8,8]
        x1_add = x1_1 * self.xishu1 + x1_2 * self.xishu2  # [B,32,8,8]
        
        # Block 2: Weight-shared processing + fusion
        x2_1 = self.conv2_1(x1_1)  # [B,64,4,4]
        x2_2 = self.conv2_1(x1_2)  # [B,64,4,4]
        x2_add = x2_1 * self.xishu1 + x2_2 * self.xishu2  # [B,64,4,4]
        
        # Block 3: Weight-shared processing + fusion
        x3_1 = self.conv3_1(x2_1)  # [B,128,2,2]
        x3_2 = self.conv3_1(x2_2)  # [B,128,2,2]
        x3_add = x3_1 * self.xishu1 + x3_2 * self.xishu2  # [B,128,2,2]
        
        return x1_add, x2_add, x3_add


class Decoder(nn.Module):
    """
    Dual-branch Decoder for Modality-specific Reconstruction
    
    Reconstructs HSI and LiDAR images from fused features for adversarial learning.
    First two blocks are weight-shared, last block is modality-specific.
    
    Input: x1_cat [batch, 128, 2, 2] (fused high-level features)
    Outputs:
        x_H: Reconstructed HSI [batch, l1, 16, 16]
        x_L: Reconstructed LiDAR [batch, l2, 16, 16]
    """
    def __init__(self, l1, l2):
        """
        Args:
            l1: Number of HSI spectral bands (reconstruction target)
            l2: Number of LiDAR bands (reconstruction target)
        """
        super(Decoder, self).__init__()

        # First two blocks: Weight-shared decoding
        self.dconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 2×2 → 4×4
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.dconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 4×4 → 8×8
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # Third block: Modality-specific reconstruction
        self.dconv3_H = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 8×8 → 16×16
            nn.Conv2d(32, l1, 3, 1, 1),  # Reconstruct HSI bands
            nn.Sigmoid(),  # Normalize to [0,1]
        )
        self.dconv3_L = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 8×8 → 16×16
            nn.Conv2d(32, l2, 3, 1, 1),  # Reconstruct LiDAR bands
            nn.Sigmoid(),  # Normalize to [0,1]
        )

    def forward(self, x1_cat):
        x = self.dconv1(x1_cat)  # [B,64,4,4]
        x = self.dconv2(x)       # [B,32,8,8]
        x_H = self.dconv3_H(x)   # [B,l1,16,16]
        x_L = self.dconv3_L(x)   # [B,l2,16,16]
        return x_H, x_L


class Classifier(nn.Module):
    """
    Multi-level Feature Fusion Classifier
    
    Performs classification using features from three different levels:
    - High-level: Semantic information [B,128,2,2]
    - Mid-level: Structural information [B,64,4,4] 
    - Low-level: Detailed information [B,32,8,8]
    
    Uses adaptive weighting to combine predictions from all levels.
    
    Inputs:
        x1: High-level features [B,128,2,2]
        x2: Mid-level features [B,64,4,4]
        x3: Low-level features [B,32,8,8]
    
    Outputs:
        Three classification probabilities [B, Classes] for each level
    """
    def __init__(self, Classes):
        """
        Args:
            Classes: Number of output classes for classification
        """
        super(Classifier, self).__init__()

        # High-level feature processing pathway
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, 1),  # [B,64,2,2]
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 1),   # [B,32,2,2]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # [B,32,1,1]
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, Classes, 1),  # [B,Classes,1,1]
        )
        
        # Mid-level feature processing pathway
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(64, 32, 1),   # [B,32,4,4]
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 16, 1),   # [B,16,4,4]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # [B,16,1,1]
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(16, Classes, 1),  # [B,Classes,1,1]
        )
        
        # Low-level feature processing pathway
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(32, 16, 1),   # [B,16,8,8]
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(16, 16, 1),   # [B,16,8,8]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # [B,16,1,1]
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(16, Classes, 1),  # [B,Classes,1,1]
        )
        
        # Learnable fusion coefficients
        self.coefficient1 = torch.nn.Parameter(torch.Tensor([0.31]))  # High-level weight
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([0.33]))  # Mid-level weight
        self.coefficient3 = torch.nn.Parameter(torch.Tensor([0.36]))  # Low-level weight

    def forward(self, x1, x2, x3):
        # Process high-level features [B,128,2,2] → [B,Classes]
        x1_1 = self.conv1(x1)    # [B,64,2,2]
        x1_2 = self.conv2(x1_1)  # [B,32,1,1]
        x1_3 = self.conv3(x1_2)  # [B,Classes,1,1]
        x1_3 = x1_3.view(x1_3.size(0), -1)  # [B,Classes]
        x1_out = F.softmax(x1_3, dim=1)     # [B,Classes]

        # Process mid-level features [B,64,4,4] → [B,Classes]
        x2_1 = self.conv1_2(x2)  # [B,32,4,4]
        x2_2 = self.conv2_2(x2_1)  # [B,16,1,1]
        x2_3 = self.conv3_2(x2_2)  # [B,Classes,1,1]
        x2_3 = x2_3.view(x2_3.size(0), -1)  # [B,Classes]
        x2_out = F.softmax(x2_3, dim=1)     # [B,Classes]

        # Process low-level features [B,32,8,8] → [B,Classes]
        x3_1 = self.conv1_3(x3)  # [B,16,8,8]
        x3_2 = self.conv2_3(x3_1)  # [B,16,1,1]
        x3_3 = self.conv3_3(x3_2)  # [B,Classes,1,1]
        x3_3 = x3_3.view(x3_3.size(0), -1)  # [B,Classes]
        x3_out = F.softmax(x3_3, dim=1)     # [B,Classes]

        return x1_out, x2_out, x3_out


class Discriminator_H(nn.Module):
    """
    HSI Discriminator with Spatial Attention
    
    Distinguishes between real HSI patches and generated HSI patches.
    Uses spatial attention to focus on discriminative regions.
    
    Input: HSI patches [batch, l1, 16, 16]
    Output: Real/fake probability [batch, 1]
    """
    def __init__(self, l1, Classes):
        super(Discriminator_H, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(l1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, Classes)
        self.sa = SpatialAttention()

    def forward(self, input):
        input = self.sa(input) * input  # Apply spatial attention
        x1 = self.conv1(input)  # [B,32,16,16]
        x2 = self.conv2(x1)     # [B,32,8,8]
        x3 = self.conv3(x2)     # [B,64,8,8]
        x4 = self.conv4(x3)     # [B,64,4,4]
        x5 = self.conv5(x4)     # [B,128,4,4]
        x6 = self.conv6(x5)     # [B,128,2,2]
        x7 = self.avgpool(x6)   # [B,128,1,1]
        x8 = x7.view(x7.size(0), -1)  # [B,128]
        x9 = self.fc(x8)        # [B,1]
        x10 = torch.sigmoid(x9) # [B,1]
        return x10


class Discriminator_L(nn.Module):
    """
    LiDAR Discriminator with Spatial Attention
    
    Distinguishes between real LiDAR patches and generated LiDAR patches.
    Uses spatial attention to focus on discriminative regions.
    
    Input: LiDAR patches [batch, l2, 16, 16]
    Output: Real/fake probability [batch, 1]
    """
    def __init__(self, l2, Classes):
        super(Discriminator_L, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(l2, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, Classes)
        self.sa = SpatialAttention()

    def forward(self, input):
        input = self.sa(input) * input  # Apply spatial attention
        x1 = self.conv1(input)  # [B,32,16,16]
        x2 = self.conv2(x1)     # [B,32,8,8]
        x3 = self.conv3(x2)     # [B,64,8,8]
        x4 = self.conv4(x3)     # [B,64,4,4]
        x5 = self.conv5(x4)     # [B,128,4,4]
        x6 = self.conv6(x5)     # [B,128,2,2]
        x7 = self.avgpool(x6)   # [B,128,1,1]
        x8 = x7.view(x7.size(0), -1)  # [B,128]
        x9 = self.fc(x8)        # [B,1]
        x10 = torch.sigmoid(x9) # [B,1]
        return x10


class Network(nn.Module):
    """
    Main CALC Network integrating all components
    
    Combines encoder, decoder, and classifier for end-to-end multimodal
    feature learning and classification.
    
    Inputs:
        x1: HSI patches [batch, l1, 16, 16]
        x2: LiDAR patches [batch, l2, 16, 16]
    
    Outputs:
        rx_H: Reconstructed HSI [batch, l1*16*16]
        rx_L: Reconstructed LiDAR [batch, l2*16*16] 
        cx1: High-level classification [batch, Classes]
        cx2: Mid-level classification [batch, Classes]
        cx3: Low-level classification [batch, Classes]
    """
    def __init__(self, l1, l2, Classes):
        """
        Args:
            l1: Number of HSI spectral bands
            l2: Number of LiDAR bands  
            Classes: Number of classification categories
        """
        super(Network, self).__init__()
        self.encoder = Encoder(l1=l1, l2=l2)
        self.decoder = Decoder(l1=l1, l2=l2)
        self.classifier = Classifier(Classes=Classes)

    def forward(self, x1, x2):
        # Feature extraction and fusion
        ex1, ex2, ex3 = self.encoder(x1, x2)  # Multi-level features
        
        # Image reconstruction for adversarial learning
        rx_H, rx_L = self.decoder(ex3)  # Reconstructed images
        
        # Multi-level classification
        cx1, cx2, cx3 = self.classifier(ex3, ex2, ex1)  # Classification probabilities
        
        # Flatten reconstructed images for discriminators
        rx_H = rx_H.view(rx_H.size(0), -1)
        rx_L = rx_L.view(rx_L.size(0), -1)
        
        return rx_H, rx_L, cx1, cx2, cx3

def train_network(train_loader, 
                  TrainPatch1, 
                  TrainPatch2, 
                  TrainLabel1, 
                  TestPatch1, 
                  TestPatch2, 
                  TestLabel, 
                  LR, 
                  EPOCH, 
                  patchsize, 
                  l1, 
                  l2, 
                  Classes, 
                  model_save_path='CALC_Trento.pkl'):  # Add model_save_path parameter
    """
    Training procedure for CALC model with adversarial learning
    
    Implements the coupled adversarial training between generators and discriminators
    with multi-level feature fusion classification.
    
    Args:
        train_loader: DataLoader for training patches
        TrainPatch1: HSI training patches
        TrainPatch2: LiDAR training patches  
        TrainLabel1: Training labels
        TestPatch1: HSI test patches
        TestPatch2: LiDAR test patches
        TestLabel: Test labels
        LR: Learning rate
        EPOCH: Number of training epochs
        patchsize: Spatial size of input patches (e.g., 16)
        l1: Number of HSI bands
        l2: Number of LiDAR bands
        Classes: Number of classification categories
        model_save_path: Path to save/load the model
    
    Returns:
        pred_y: Final predictions on test set
        val_acc: Validation accuracy history
    """
    # Initialize networks
    cnn = Network(l1=l1, l2=l2, Classes=Classes)
    dis_H = Discriminator_H(l1=l1, Classes=Classes)
    dis_L = Discriminator_L(l2=l2, Classes=Classes)
    
    # Move to GPU
    cnn.cuda()
    dis_H.cuda()
    dis_L.cuda()
    
    # Optimizers
    g_optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # Generator optimizer
    d_optimizer_H = torch.optim.Adam(dis_H.parameters(), lr=LR)  # HSI discriminator
    d_optimizer_L = torch.optim.Adam(dis_L.parameters(), lr=LR)  # LiDAR discriminator
    
    # Loss functions
    loss_fun1 = nn.CrossEntropyLoss()  # Classification loss
    loss_fun2 = nn.MSELoss()  # Feature consistency loss
    
    val_acc = []
    class_loss = []
    gan_loss = []
    BestAcc = 0
    
    # Training loop
    for epoch in range(EPOCH):
        for step, (b_x1, b_x2, b_y) in enumerate(train_loader):
            # Move batch to GPU
            b_x1 = b_x1.cuda()  # [64,144,16,16]
            b_x2 = b_x2.cuda()  # [64,1,16,16]
            b_y = b_y.cuda()    # [64]
            
            # Forward pass
            fake_H, fake_L, output, output2, output3 = cnn(b_x1, b_x2)
            
            # Train HSI discriminator
            dis_H.zero_grad()
            fake_probability = dis_H(fake_H.view(fake_H.size(0), l1, patchsize, patchsize))
            fake_probability = fake_probability.mean()
            real_probability = dis_H(b_x1)
            real_probability = real_probability.mean()
            d_loss1 = 1 - real_probability + fake_probability  # Wasserstein loss
            d_loss1.backward(retain_graph=True)
            
            # Train LiDAR discriminator  
            dis_L.zero_grad()
            fake_probability2 = dis_L(fake_L.view(fake_L.size(0), l2, patchsize, patchsize))
            fake_probability2 = fake_probability2.mean()
            real_probability2 = dis_L(b_x2)
            real_probability2 = real_probability2.mean()
            d_loss2 = 1 - real_probability2 + fake_probability2  # Wasserstein loss
            d_loss2.backward(retain_graph=True)
            
            # Train generator (encoder + decoder + classifier)
            cnn.zero_grad()
            ce_loss = loss_fun1(output, b_y.long()) + loss_fun2(output, output2) + loss_fun2(output, output3)
            a_loss = 0.5 * torch.mean(1 - fake_probability) + 0.5 * torch.mean(1 - fake_probability2)
            g_loss = 0.01 * a_loss + ce_loss  # Combined loss with adversarial weight
            g_loss.backward()
            
            # Update optimizers
            d_optimizer_H.step()
            d_optimizer_L.step()
            g_optimizer.step()

            # Validation and logging
            if step % 50 == 0:
                cnn.eval()
                # Calculate adaptive weights based on training performance
                temp1 = TrainPatch1.cuda()
                temp2 = TrainPatch2.cuda()
                _, _, temp3, temp4, temp5 = cnn(temp1, temp2)
                
                # Calculate accuracy for each level
                pred_y1 = torch.max(temp3, 1)[1].squeeze().cpu()
                acc1 = torch.sum(pred_y1 == TrainLabel1).type(torch.FloatTensor) / TrainLabel1.size(0)
                pred_y2 = torch.max(temp4, 1)[1].squeeze().cpu()
                acc2 = torch.sum(pred_y2 == TrainLabel1).type(torch.FloatTensor) / TrainLabel1.size(0)
                pred_y3 = torch.max(temp5, 1)[1].squeeze().cpu()
                acc3 = torch.sum(pred_y3 == TrainLabel1).type(torch.FloatTensor) / TrainLabel1.size(0)
                
                # Calculate class-wise weights for fusion
                Classes = np.unique(TrainLabel1)
                w0 = np.empty(len(Classes), dtype='float32')
                w1 = np.empty(len(Classes), dtype='float32')
                w2 = np.empty(len(Classes), dtype='float32')
                
                for i in range(len(Classes)):
                    cla = Classes[i]
                    right1 = right2 = right3 = 0
                    
                    for j in range(len(TrainLabel1)):
                        if TrainLabel1[j] == cla:
                            if pred_y1[j] == cla: right1 += 1
                            if pred_y2[j] == cla: right2 += 1  
                            if pred_y3[j] == cla: right3 += 1
                    
                    total = right1 + right2 + right3 + 0.00001
                    w0[i] = right1 / total
                    w1[i] = right2 / total
                    w2[i] = right3 / total

                w0 = torch.from_numpy(w0).cuda()
                w1 = torch.from_numpy(w1).cuda()
                w2 = torch.from_numpy(w2).cuda()

                # Test set evaluation
                pred_y = np.empty((len(TestLabel)), dtype='float32')
                number = len(TestLabel) // 100
                
                for i in range(number):
                    temp1_1 = TestPatch1[i * 100:(i + 1) * 100, :, :, :].cuda()
                    temp1_2 = TestPatch2[i * 100:(i + 1) * 100, :, :, :].cuda()
                    temp2 = w2 * cnn(temp1_1, temp1_2)[4] + w1 * cnn(temp1_1, temp1_2)[3] + w0 * cnn(temp1_1, temp1_2)[2]
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[i * 100:(i + 1) * 100] = temp3.cpu()

                if (i + 1) * 100 < len(TestLabel):
                    temp1_1 = TestPatch1[(i + 1) * 100:len(TestLabel), :, :, :].cuda()
                    temp1_2 = TestPatch2[(i + 1) * 100:len(TestLabel), :, :, :].cuda()
                    temp2 = w2 * cnn(temp1_1, temp1_2)[4] + w1 * cnn(temp1_1, temp1_2)[3] + w0 * cnn(temp1_1, temp1_2)[2]
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[(i + 1) * 100:len(TestLabel)] = temp3.cpu()

                pred_y = torch.from_numpy(pred_y).long()
                accuracy = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
                
                print('Epoch: ', epoch, '| classify loss: %.6f' % ce_loss.data.cpu().numpy(),
                      '| test accuracy: %.6f' % accuracy, '| w0: %.2f' % w0[0], '| w1: %.2f' % w1[0],
                      '| w2: %.2f' % w2[0])
                
                val_acc.append(accuracy.data.cpu().numpy())
                class_loss.append(ce_loss.data.cpu().numpy())
                gan_loss.append(g_loss.data.cpu().numpy())
                
                # Save best model - use the provided model_save_path
                if accuracy > BestAcc:
                    torch.save(cnn.state_dict(), model_save_path)  # Use parameter instead of hardcoded name
                    BestAcc = accuracy
                    w0B = w0
                    w1B = w1
                    w2B = w2
                
                cnn.train()  # Return to training mode

    # Final evaluation with best model - use the provided model_save_path
    cnn.load_state_dict(torch.load(model_save_path))  # Use parameter instead of hardcoded name
    cnn.eval()
    w0, w1, w2 = w0B, w1B, w2B

    # Generate final predictions
    pred_y = np.empty((len(TestLabel)), dtype='float32')
    number = len(TestLabel) // 100
    
    for i in range(number):
        temp1_1 = TestPatch1[i * 100:(i + 1) * 100, :, :, :].cuda()
        temp1_2 = TestPatch2[i * 100:(i + 1) * 100, :, :, :].cuda()
        temp2 = w2 * cnn(temp1_1, temp1_2)[4] + w1 * cnn(temp1_1, temp1_2)[3] + w0 * cnn(temp1_1, temp1_2)[2]
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[i * 100:(i + 1) * 100] = temp3.cpu()

    if (i + 1) * 100 < len(TestLabel):
        temp1_1 = TestPatch1[(i + 1) * 100:len(TestLabel), :, :, :].cuda()
        temp1_2 = TestPatch2[(i + 1) * 100:len(TestLabel), :, :, :].cuda()
        temp2 = w2 * cnn(temp1_1, temp1_2)[4] + w1 * cnn(temp1_1, temp1_2)[3] + w0 * cnn(temp1_1, temp1_2)[2]
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[(i + 1) * 100:len(TestLabel)] = temp3.cpu()

    pred_y = torch.from_numpy(pred_y).long()
    
    # Calculate final metrics
    OA = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
    Classes = np.unique(TestLabel)
    EachAcc = np.empty(len(Classes))

    for i in range(len(Classes)):
        cla = Classes[i]
        right = sum_count = 0
        for j in range(len(TestLabel)):
            if TestLabel[j] == cla:
                sum_count += 1
                if pred_y[j] == cla:
                    right += 1
        EachAcc[i] = right / sum_count if sum_count > 0 else 0

    AA = np.mean(EachAcc)

    print(f"Final OA: {OA:.4f}, AA: {AA:.4f}")
    print(f"Per-class Accuracies: {EachAcc}")
    
    return pred_y, val_acc