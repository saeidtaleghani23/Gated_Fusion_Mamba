import torch
import torch.nn as nn
import sys
import os

# Add the current directory to path to import from other files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from UnifiedFusionLayer import UnifiedFusionLayer


class MultiLayerFusionClassifier(nn.Module):
    """
    Complete multimodal fusion classifier with multiple UnifiedFusionLayer blocks
    and classification head for patch-level classification
    """
    
    def __init__(self, 
                 hsi_channels=138,
                 lidar_channels=1,
                 feature_dims=None,  # ← Changed to list for per-layer dimensions
                 num_layers=3,
                 num_classes=10,
                 use_downsampling=True,
                 downsample_method='conv',
                 layer_clusters=None,
                 patch_size=32,
                 mamba_config=None):
        
        super(MultiLayerFusionClassifier, self).__init__()
        
        # Set defaults if not provided
        if feature_dims is None:
            feature_dims = [64] * num_layers  # Default to 64 for all layers
        
        if layer_clusters is None:
            layer_clusters = [50, 25, 10]
        
        # Validate inputs
        if len(feature_dims) != num_layers:
            raise ValueError(f"feature_dims length ({len(feature_dims)}) must match num_layers ({num_layers})")
        
        if len(layer_clusters) != num_layers:
            raise ValueError(f"layer_clusters must have length {num_layers}, got {len(layer_clusters)}")
        
        # Configuration - STORE ALL CONFIG
        self.config = {
            'hsi_channels': hsi_channels,
            'lidar_channels': lidar_channels,
            'feature_dims': feature_dims,  # ← Now stores list
            'num_layers': num_layers,
            'num_classes': num_classes,
            'use_downsampling': use_downsampling,
            'downsample_method': downsample_method,
            'layer_clusters': layer_clusters,
            'patch_size': patch_size,
            'mamba_config': mamba_config
        }
        
        # Default Mamba configuration
        if mamba_config is None:
            mamba_config = {
                'd_state': 16,
                'd_conv': 4,
                'expand': 2
            }
        
        # ===== 1. MULTI-LAYER FUSION BLOCKS =====
        self.fusion_layers = nn.ModuleList()
        
        # Track input channels for each layer
        current_hsi_channels = hsi_channels
        current_lidar_channels = lidar_channels
        
        for i in range(num_layers):
            fusion_layer = UnifiedFusionLayer(
                hsi_channels=current_hsi_channels,
                lidar_channels=current_lidar_channels,
                feature_dim=feature_dims[i],  # ← Use per-layer feature dimension
                hsi_clusters=layer_clusters[i],
                lidar_clusters=layer_clusters[i],
                use_downsampling=use_downsampling,
                downsample_method=downsample_method,
                mamba_config=mamba_config
            )
            self.fusion_layers.append(fusion_layer)
            
            # Update input channels for next layer
            current_hsi_channels = feature_dims[i]
            current_lidar_channels = feature_dims[i]
        
        # Store final feature dimension for classification head
        self.final_feature_dim = feature_dims[-1]
        
        # ===== 2. FEATURE FUSION AND CLASSIFICATION HEAD =====
        # Calculate final spatial dimensions after all layers
        final_spatial = self._calculate_final_spatial(patch_size, num_layers, use_downsampling)
        self.final_height, self.final_width = final_spatial
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head - uses final feature dimension
        # Multiply by 2 because we concatenate D_hsi and D_lidar
        classifier_input_dim = self.final_feature_dim * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.BatchNorm1d(128),  # ← ADD THIS
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        #self.classifier = nn.Sequential(
            #nn.Linear(classifier_input_dim, num_classes),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            # nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            # nn.Linear(256, num_classes)
        # )
        
        # Initialize weights
        self._initialize_weights()
    
    def get_config(self):
        """Return model configuration"""
        return self.config.copy()
    
    def _calculate_final_spatial(self, patch_size, num_layers, use_downsampling):
        """Calculate final spatial dimensions after all layers"""
        if not use_downsampling:
            return patch_size, patch_size
        
        # Each downsampling layer reduces by factor of 2
        final_size = patch_size
        for _ in range(num_layers):
            final_size = final_size // 2
        
        return final_size, final_size
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, hsi_patch, lidar_patch):
        """
        Forward pass through multiple fusion layers and classification head
        """
        batch_size = hsi_patch.shape[0]
        
        # Store outputs from each layer for analysis
        layer_outputs = []
        
        # Current features for each modality
        current_hsi = hsi_patch
        current_lidar = lidar_patch
        
        # ===== PROCESS THROUGH MULTIPLE FUSION LAYERS =====
        for i, fusion_layer in enumerate(self.fusion_layers):
            # Apply fusion layer
            D_hsi, D_lidar = fusion_layer(current_hsi, current_lidar)
            
            # Store layer outputs
            layer_outputs.append((D_hsi, D_lidar))
            
            # Update current features for next layer
            current_hsi = D_hsi
            current_lidar = D_lidar
        
        # Final enhanced features from last layer
        final_D_hsi, final_D_lidar = layer_outputs[-1]
        
        # ===== FEATURE FUSION FOR CLASSIFICATION =====
        # Concatenate final enhanced features
        fused_features = torch.cat([final_D_hsi, final_D_lidar], dim=1)
        
        # Global average pooling
        global_features = self.global_pool(fused_features)
        global_features = global_features.view(batch_size, -1)
        
        # Classification
        logits = self.classifier(global_features)
        
        return logits, layer_outputs
    
    def get_layer_info(self):
        """Get information about each layer's configuration"""
        layer_info = []
        for i, (fusion_layer, num_clusters, feature_dim) in enumerate(
            zip(self.fusion_layers, self.config['layer_clusters'], self.config['feature_dims'])):
            
            info = {
                'layer': i + 1,
                'clusters': num_clusters,
                'feature_dim': feature_dim,  # ← Include feature dimension
                'downsampling': fusion_layer.use_downsampling,
                'input_channels_hsi': fusion_layer.hsi_channels,
                'input_channels_lidar': fusion_layer.lidar_channels
            }
            layer_info.append(info)
        return layer_info
    
    def set_downsampling(self, enabled, layer_idx=None):
        """
        Enable/disable downsampling for ablation studies
        """
        if layer_idx is None:
            # Apply to all layers
            for fusion_layer in self.fusion_layers:
                fusion_layer.set_downsampling(enabled)
        else:
            # Apply to specific layer
            self.fusion_layers[layer_idx].set_downsampling(enabled)


# Default configuration for your specific requirements
DEFAULT_CONFIG = {
    'hsi_channels': 138,
    'lidar_channels': 1,
    'feature_dims': [64, 64, 64],  # Updated to use feature_dims instead of feature_dim
    'num_layers': 3,
    'num_classes': 10,  # Adjust based on your classification task
    'use_downsampling': True,
    'downsample_method': 'conv',
    'layer_clusters': [50, 25, 10],  # First layer: 50, Second: 25, Third: 10 clusters
    'patch_size': 32
}


def create_model(config=None):
    """Convenience function to create model with default or custom config"""
    if config is None:
        config = DEFAULT_CONFIG
    
    return MultiLayerFusionClassifier(**config)


# Test the complete model
if __name__ == "__main__":
    print("Testing MultiLayerFusionClassifier")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with default config (your requirements)
    model = create_model().to(device)
    
    # Test input (32x32 patches as specified)
    batch_size = 4
    hsi_input = torch.randn(batch_size, 138, 32, 32).to(device)
    lidar_input = torch.randn(batch_size, 1, 32, 32).to(device)
    
    print(f"Input - HSI: {hsi_input.shape}, LiDAR: {lidar_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits, layer_outputs = model(hsi_input, lidar_input)
    
    print(f"\nModel Output:")
    print(f"Logits shape: {logits.shape}")  # [4, 10]
    print(f"Number of layers: {len(layer_outputs)}")
    
    # Print layer information
    print(f"\nLayer Configuration:")
    layer_info = model.get_layer_info()
    for info in layer_info:
        print(f"Layer {info['layer']}: {info['clusters']} clusters, "
              f"Feature Dim: {info['feature_dim']}, "
              f"Downsampling: {info['downsampling']}, "
              f"Input: HSI{info['input_channels_hsi']}/LiDAR{info['input_channels_lidar']}")
    
    # Print spatial dimensions through layers
    print(f"\nSpatial Dimensions Through Layers:")
    print(f"Input: 32x32")
    for i, (D_hsi, D_lidar) in enumerate(layer_outputs):
        print(f"Layer {i+1}: {D_hsi.shape[2]}x{D_hsi.shape[3]}")
    
    # Test downsampling ablation
    print(f"\nTesting Downsampling Ablation:")
    model_no_ds = create_model({'use_downsampling': False}).to(device)
    with torch.no_grad():
        logits_no_ds, layer_outputs_no_ds = model_no_ds(hsi_input, lidar_input)
    print(f"With downsampling: {logits.shape}")
    print(f"Without downsampling: {logits_no_ds.shape}")
    
    # Test dynamic switching
    print(f"\nTesting Dynamic Layer Switching:")
    model_dynamic = create_model().to(device)
    
    # Disable downsampling in middle layer only
    model_dynamic.set_downsampling(False, layer_idx=1)
    with torch.no_grad():
        logits_dynamic, _ = model_dynamic(hsi_input, lidar_input)
    print(f"Mixed downsampling (Layer 2 disabled): {logits_dynamic.shape}")
    
    # Model summary
    print(f"\nModel Summary:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print(f"\n✅ MultiLayerFusionClassifier is ready for training!")