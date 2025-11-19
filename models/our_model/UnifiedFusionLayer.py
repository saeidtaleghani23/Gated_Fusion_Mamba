import torch
import torch.nn as nn
import sys
import os

# Add the current directory to path to import from other files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all required components
from ClusterMamba import ModalityBranch, ClusterMambaLayer
from DynamicFeatureFusion import HSI_LiDAR_Fusion, BranchFusion, LDC
from DownsamplingProjection import OptionalDownsampling

import torch
import torch.nn as nn
import sys
import os

# Add the current directory to path to import from other files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all required components
from ClusterMamba import ModalityBranch, ClusterMambaLayer
from DynamicFeatureFusion import HSI_LiDAR_Fusion, BranchFusion, LDC
from DownsamplingProjection import OptionalDownsampling

class UnifiedFusionLayer(nn.Module):
    """
    Complete fusion layer that integrates:
    - Optional downsampling (using our existing DownsamplingProjection)
    - Modality-specific branches (HSI/LiDAR) with ClusterMamba
    - Dynamic feature fusion
    """
    
    def __init__(self, 
                 hsi_channels=138, 
                 lidar_channels=1, 
                 feature_dim=64,
                 hsi_clusters=8,
                 lidar_clusters=6,
                 use_downsampling=True,
                 downsample_method='conv',
                 mamba_config=None):
        
        super(UnifiedFusionLayer, self).__init__()
        
        # Configuration
        self.feature_dim = feature_dim
        self.use_downsampling = use_downsampling
        self.hsi_channels = hsi_channels
        self.lidar_channels = lidar_channels
        self.downsample_method = downsample_method
        
        # Default Mamba configuration
        if mamba_config is None:
            mamba_config = {
                'd_state': 16,
                'd_conv': 4,
                'expand': 2
            }
        
        # ===== 1. OPTIONAL DOWNSAMPLING =====
        # Use our existing OptionalDownsampling class
        self.downsample_hsi = OptionalDownsampling(
            hsi_channels, feature_dim, method=downsample_method, enabled=use_downsampling
        )
        self.downsample_lidar = OptionalDownsampling(
            lidar_channels, feature_dim, method=downsample_method, enabled=use_downsampling
        )
        
        # ===== 2. MODALITY BRANCHES =====
        # Based on the error, ModalityBranch takes 3 arguments: input_channels, feature_dim, num_clusters
        self.hsi_branch = ModalityBranch(
            input_channels=feature_dim, 
            feature_dim=feature_dim, 
            num_clusters=hsi_clusters
        )
        self.lidar_branch = ModalityBranch(
            input_channels=feature_dim, 
            feature_dim=feature_dim, 
            num_clusters=lidar_clusters
        )
        
        # ===== 3. BATCH NORMALIZATION LAYERS =====
        # Add BN after modality branches to stabilize features before fusion
        self.bn_hsi = nn.BatchNorm2d(feature_dim)  # ← ADD THIS
        self.bn_lidar = nn.BatchNorm2d(feature_dim)  # ← ADD THIS
        
        # ===== 4. DYNAMIC FUSION =====
        # Use existing HSI_LiDAR_Fusion
        self.fusion = HSI_LiDAR_Fusion(feature_dim)
        
        # ===== 5. POST-FUSION BATCH NORM =====
        # Add BN after fusion to stabilize the final outputs
        self.bn_hsi_final = nn.BatchNorm2d(feature_dim)  # ← ADD THIS
        self.bn_lidar_final = nn.BatchNorm2d(feature_dim)  # ← ADD THIS
    
    def forward(self, hsi_patch, lidar_patch):
        """
        Process HSI and LiDAR patches through the complete pipeline
        
        Args:
            hsi_patch: [B, 138, H, W] 
            lidar_patch: [B, 1, H, W]
            
        Returns:
            D_hsi: Enhanced HSI features [B, 64, H_current, W_current]
            D_lidar: Enhanced LiDAR features [B, 64, H_current, W_current]
        """
        # ===== STEP 1: DOWNSAMPLING =====
        # Apply our existing downsampling (reduces spatial size if enabled)
        hsi_down = self.downsample_hsi(hsi_patch)  # [B, 64, H/2, W/2] if enabled, else [B, 64, H, W]
        lidar_down = self.downsample_lidar(lidar_patch)  # [B, 64, H/2, W/2] if enabled, else [B, 64, H, W]
        
        # ===== STEP 2: MODALITY-SPECIFIC PROCESSING =====
        # Each branch applies its own ClusterMamba processing
        hsi_features = self.hsi_branch(hsi_down)  # [B, 64, H_current, W_current]
        lidar_features = self.lidar_branch(lidar_down)  # [B, 64, H_current, W_current]
        
        # ===== STEP 3: BATCH NORMALIZATION BEFORE FUSION =====
        # Stabilize features before fusion
        hsi_features = self.bn_hsi(hsi_features)  # ← ADD THIS
        lidar_features = self.bn_lidar(lidar_features)  # ← ADD THIS
        
        # ===== STEP 4: DYNAMIC FUSION =====
        # Apply the fusion mechanism from DynamicFeatureFusion
        D_hsi, D_lidar = self.fusion(hsi_features, lidar_features)  # [B, 64, H_current, W_current]
        
        # ===== STEP 5: BATCH NORMALIZATION AFTER FUSION =====
        # Stabilize final outputs
        D_hsi = self.bn_hsi_final(D_hsi)  # ← ADD THIS
        D_lidar = self.bn_lidar_final(D_lidar)  # ← ADD THIS
        
        return D_hsi, D_lidar

    def set_downsampling(self, enabled):
        """Dynamically enable/disable downsampling for ablation studies"""
        # Store current device
        current_device = next(self.parameters()).device
        
        # Enable/disable downsampling
        self.downsample_hsi.set_enabled(enabled)
        self.downsample_lidar.set_enabled(enabled)
        self.use_downsampling = enabled
        
        # Ensure all components are on the same device
        self.downsample_hsi = self.downsample_hsi.to(current_device)
        self.downsample_lidar = self.downsample_lidar.to(current_device)


# Test the complete layer with proper GPU handling
if __name__ == "__main__":
    print("Testing UnifiedFusionLayer")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size, h, w = 2, 128, 128
    hsi_channels, lidar_channels = 138, 1
    
    # Test with downsampling enabled
    print("With Downsampling:")
    fusion_layer = UnifiedFusionLayer(
        hsi_channels=hsi_channels,
        lidar_channels=lidar_channels,
        use_downsampling=True,
        downsample_method='conv'
    ).to(device)  # Move model to GPU
    
    hsi_input = torch.randn(batch_size, hsi_channels, h, w).to(device)  # Move input to GPU
    lidar_input = torch.randn(batch_size, lidar_channels, h, w).to(device)  # Move input to GPU
    
    D_hsi, D_lidar = fusion_layer(hsi_input, lidar_input)
    print(f"Input - HSI: {hsi_input.shape}, LiDAR: {lidar_input.shape}")
    print(f"Output - D_hsi: {D_hsi.shape}, D_lidar: {D_lidar.shape}")
    print(f"Output device - D_hsi: {D_hsi.device}, D_lidar: {D_lidar.device}")
    
    # Test with downsampling disabled
    print("\nWithout Downsampling:")
    fusion_layer_no_ds = UnifiedFusionLayer(
        hsi_channels=hsi_channels,
        lidar_channels=lidar_channels,
        use_downsampling=False
    ).to(device)  # Move model to GPU
    
    D_hsi_no_ds, D_lidar_no_ds = fusion_layer_no_ds(hsi_input, lidar_input)
    print(f"Input - HSI: {hsi_input.shape}, LiDAR: {lidar_input.shape}")
    print(f"Output - D_hsi: {D_hsi_no_ds.shape}, D_lidar: {D_lidar_no_ds.shape}")
    
    # Test dynamic switching - FIXED VERSION
    print("\nTesting Dynamic Switching (Fixed):")
    fusion_dynamic = UnifiedFusionLayer(use_downsampling=True).to(device)
    
    # Initially with downsampling
    D_hsi1, D_lidar1 = fusion_dynamic(hsi_input, lidar_input)
    print(f"With downsampling: {hsi_input.shape} -> {D_hsi1.shape}")
    
    # Switch to no downsampling - now properly handles device
    fusion_dynamic.set_downsampling(False)
    D_hsi2, D_lidar2 = fusion_dynamic(hsi_input, lidar_input)
    print(f"Without downsampling: {hsi_input.shape} -> {D_hsi2.shape}")
    
    # Switch back to downsampling
    fusion_dynamic.set_downsampling(True)
    D_hsi3, D_lidar3 = fusion_dynamic(hsi_input, lidar_input)
    print(f"Back to downsampling: {hsi_input.shape} -> {D_hsi3.shape}")
    
    # Test different patch sizes
    print("\nTesting different patch sizes:")
    sizes = [(64, 64), (128, 128), (256, 256)]
    
    for h_test, w_test in sizes:
        hsi_test = torch.randn(1, hsi_channels, h_test, w_test).to(device)
        lidar_test = torch.randn(1, lidar_channels, h_test, w_test).to(device)
        
        fusion_test = UnifiedFusionLayer(use_downsampling=True).to(device)
        D_hsi_test, D_lidar_test = fusion_test(hsi_test, lidar_test)
        
        input_spatial = f"{h_test}x{w_test}"
        output_spatial = f"{D_hsi_test.shape[2]}x{D_hsi_test.shape[3]}"
        reduction = f"{h_test/D_hsi_test.shape[2]:.1f}x"
        
        print(f"Size {input_spatial:8s} -> {output_spatial:8s} (reduction: {reduction})")
    
    print("\n✅ All tests passed! UnifiedFusionLayer is working correctly.")