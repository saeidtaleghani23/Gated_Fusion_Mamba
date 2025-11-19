import torch
import torch.nn as nn
import torch.nn.functional as F

class BranchFusion(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super(BranchFusion, self).__init__()
        
        # LDC modules for each branch
        self.ldc_hsi = LDC(channels, channels, kernel_size=kernel_size, padding=padding)
        self.ldc_lidar = LDC(channels, channels, kernel_size=kernel_size, padding=padding)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, F_hsi, F_lidar):
        """
        Fuse HSI and LiDAR branch outputs using Equation 7
        
        Args:
            F_hsi: HSI branch feature map [B, C, H, W]
            F_lidar: LiDAR branch feature map [B, C, H, W]
            
        Returns:
            D_hsi: Enhanced HSI features [B, C, H, W]
            D_lidar: Enhanced LiDAR features [B, C, H, W]
        """
        
        # Coarse-grained feature fusion (simple addition)
        F_fused = F_hsi + F_lidar  # F_f^n in Equation 7
        
        # Apply LDC to enhance texture information
        T_hsi = self.sigmoid(self.ldc_hsi(F_hsi))  # T_1^n
        T_lidar = self.sigmoid(self.ldc_lidar(F_lidar))  # T_2^n
        
        # Calculate difference weights
        diff_hsi_lidar = T_lidar - T_hsi  # T_2^n - T_1^n
        diff_lidar_hsi = T_hsi - T_lidar  # T_1^n - T_2^n
        
        # Global average pooling + sigmoid to get difference weights
        weight_hsi = self.sigmoid(self.gap(diff_hsi_lidar))  # δ(GAP(T_2^n - T_1^n))
        weight_lidar = self.sigmoid(self.gap(diff_lidar_hsi))  # δ(GAP(T_1^n - T_2^n))
        
        # Apply Equation 7
        D_hsi = F_hsi + T_hsi + (weight_hsi * F_fused)  # D_1^n
        D_lidar = F_lidar + T_lidar + (weight_lidar * F_fused)  # D_2^n
        
        return D_hsi, D_lidar


# Your provided LDC class (with minor fix for device handling)
class LDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(LDC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             stride=stride, padding=padding, dilation=dilation, 
                             groups=groups, bias=bias)
        
        # Register buffer instead of using .cuda() for better device handling
        self.register_buffer('center_mask', torch.tensor([[0, 0, 0],
                                                         [0, 1, 0],
                                                         [0, 0, 0]], dtype=torch.float32))
        
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)
        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0), self.conv.weight.size(1)]),
                                           requires_grad=True)
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)

    def forward(self, x):
        # Ensure center_mask is on the same device as conv weights
        center_mask = self.center_mask.to(self.conv.weight.device)
        
        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None] * \
               center_mask * self.conv.weight.sum(2).sum(2)[:, :, None, None]

        out_diff = F.conv2d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, 
                           stride=self.conv.stride, padding=self.conv.padding,
                           groups=self.conv.groups)
        return out_diff


# Complete fusion module that integrates with your existing architecture
class HSI_LiDAR_Fusion(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super(HSI_LiDAR_Fusion, self).__init__()
        
        self.branch_fusion = BranchFusion(channels, kernel_size, padding)
        
    def forward(self, F_hsi, F_lidar):
        """
        Fuse HSI and LiDAR features
        
        Args:
            F_hsi: HSI branch output features [B, C, H, W]
            F_lidar: LiDAR branch output features [B, C, H, W]
            
        Returns:
            D_hsi: Enhanced HSI features
            D_lidar: Enhanced LiDAR features
        """
        
        D_hsi, D_lidar = self.branch_fusion(F_hsi, F_lidar)
        
        return D_hsi, D_lidar


# Example usage
if __name__ == "__main__":
    # Example dimensions
    batch_size, channels, height, width = 4, 64, 128, 128
    
    # Create fusion module
    fusion_module = HSI_LiDAR_Fusion(channels=channels)
    
    # Example inputs (HSI and LiDAR branch outputs)
    F_hsi = torch.randn(batch_size, channels, height, width)
    F_lidar = torch.randn(batch_size, channels, height, width)
    
    # Fuse the features
    D_hsi, D_lidar = fusion_module(F_hsi, F_lidar)
    
    print(f"Input HSI shape: {F_hsi.shape}")
    print(f"Input LiDAR shape: {F_lidar.shape}")
    print(f"Output D_hsi shape: {D_hsi.shape}")
    print(f"Output D_lidar shape: {D_lidar.shape}")