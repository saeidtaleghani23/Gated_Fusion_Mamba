import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class DifferentiableHardKMeans(nn.Module):
    """Differentiable clustering with hard assignments using Gumbel-Softmax"""
    def __init__(self, num_clusters, feature_dim, temperature=0.1):
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.cluster_centers = nn.Parameter(torch.randn(1, num_clusters, feature_dim))
        
    def forward(self, pixel_features):
        batch_size, num_pixels, feature_dim = pixel_features.shape
        
        # Compute distances from each pixel to each cluster center
        distances = torch.cdist(pixel_features, self.cluster_centers)  # [B, N, K]
        
        # Use Gumbel-Softmax to get hard-ish assignments that are still differentiable
        cluster_assignments = F.gumbel_softmax(-distances, tau=self.temperature, hard=True, dim=-1)
        
        return cluster_assignments

class ClusterProcessor(nn.Module):
    """Processes pixels within each cluster using local Mamba"""
    def __init__(self, feature_dim, mamba_config):
        super().__init__()
        self.feature_dim = feature_dim
        self.local_mamba = Mamba(d_model=feature_dim, **mamba_config)
        self.norm = nn.LayerNorm(feature_dim)
        # REMOVED: self.bn_after_mamba = nn.BatchNorm1d(feature_dim)
        
    def forward(self, cluster_pixels):
        # Apply local Mamba to update pixel features within the cluster
        updated_pixels = self.local_mamba(cluster_pixels)
        updated_pixels = self.norm(updated_pixels)
        # REMOVED: updated_pixels = self.bn_after_mamba(updated_pixels.transpose(1, 2)).transpose(1, 2)
        
        return updated_pixels

class RepresentativeCalculator(nn.Module):
    """Calculates representative vector for each cluster using attention"""
    def __init__(self, feature_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            # REMOVED: nn.BatchNorm1d(feature_dim // 2),  # ← THIS WAS CAUSING THE ERROR
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
    def forward(self, cluster_pixels):
        batch_size, num_pixels, feature_dim = cluster_pixels.shape
        
        # Calculate attention scores for each pixel in the cluster
        attention_scores = self.attention_net(cluster_pixels)  # [B, N, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, N, 1]
        
        # Weighted average to get representative vector
        representative_vector = torch.sum(attention_weights * cluster_pixels, dim=1)  # [B, feature_dim]
        
        return representative_vector
    
class ClusterMambaLayer(nn.Module):
    """Complete cluster-based processing: cluster → local Mamba → global Mamba → fusion"""
    def __init__(self, feature_dim, num_clusters, mamba_config):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        
        # Components
        self.clustering = DifferentiableHardKMeans(num_clusters, feature_dim)
        self.cluster_processor = ClusterProcessor(feature_dim, mamba_config)
        self.representative_calculator = RepresentativeCalculator(feature_dim)
        
        # Global Mamba for processing cluster representatives
        self.global_mamba = Mamba(d_model=feature_dim, **mamba_config)
        self.global_norm = nn.LayerNorm(feature_dim)
        # REMOVED: self.bn_global = nn.BatchNorm1d(feature_dim)
        
        # === FUSION GATE DEFINITION ===
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            # REMOVED: nn.BatchNorm1d(feature_dim // 2),  # ← THIS WAS CAUSING THE ERROR
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1)  # Outputs single fusion weight
        )
        # ==============================
        
    def forward(self, all_pixel_features, image_height, image_width):
        batch_size, num_pixels, feature_dim = all_pixel_features.shape
        
        # Step 1: Cluster pixels
        cluster_assignments = self.clustering(all_pixel_features)  # [B, N, K]
        
        # Step 2: Vectorized cluster processing
        cluster_masks = cluster_assignments.transpose(1, 2)  # [B, K, N]
        cluster_sizes = cluster_masks.sum(dim=2)  # [B, K]
        
        all_representatives = []
        all_updated_pixels = []
        
        for cluster_idx in range(self.num_clusters):
            # Get mask for this cluster across all batches
            mask = cluster_masks[:, cluster_idx, :].unsqueeze(-1)  # [B, N, 1]
            cluster_size = cluster_sizes[:, cluster_idx]  # [B]
            
            # Extract and process cluster pixels
            cluster_pixels = all_pixel_features * mask  # [B, N, D]
            
            # Apply local Mamba to entire batch
            updated_cluster_pixels = self.cluster_processor(cluster_pixels)  # [B, N, D]
            
            # Calculate representative using masked attention
            attention_mask = mask.squeeze(-1).float()  # [B, N]
            attention_scores = self.representative_calculator.attention_net(updated_cluster_pixels)  # [B, N, 1]
            
            # Apply mask to attention scores
            attention_scores = attention_scores.squeeze(-1)  # [B, N]
            attention_scores = attention_scores - (1 - attention_mask) * 1e9
            attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)  # [B, N, 1]
            
            # Weighted average
            representative_vector = torch.sum(attention_weights * updated_cluster_pixels, dim=1)  # [B, D]
            
            all_representatives.append(representative_vector)
            all_updated_pixels.append(updated_cluster_pixels)
        
        # Step 3: Global Mamba on representatives
        representatives_tensor = torch.stack(all_representatives, dim=1)  # [B, K, D]
        updated_representatives = self.global_mamba(representatives_tensor)
        updated_representatives = self.global_norm(updated_representatives)
        # REMOVED: updated_representatives = self.bn_global(updated_representatives.transpose(1, 2)).transpose(1, 2)
        
        # Step 4: Vectorized reconstruction with global context
        final_pixel_features = torch.zeros_like(all_pixel_features)
        
        for cluster_idx in range(self.num_clusters):
            updated_pixels = all_updated_pixels[cluster_idx]  # [B, N, D]
            cluster_global_context = updated_representatives[:, cluster_idx]  # [B, D]
            
            # Expand global context to match pixel dimensions
            cluster_global_context_expanded = cluster_global_context.unsqueeze(1).expand(-1, num_pixels, -1)  # [B, N, D]
            
            # Calculate fusion weights for all pixels in batch
            fusion_weights = torch.sigmoid(
                self.fusion_gate(cluster_global_context_expanded)  # [B, N, 1]
            ).squeeze(-1)  # [B, N]
            
            # Get the mask for this cluster
            cluster_mask = cluster_masks[:, cluster_idx, :]  # [B, N]
            
            # Apply fusion only to pixels in this cluster
            fusion_weights_masked = fusion_weights * cluster_mask  # [B, N]
            fusion_weights_expanded = fusion_weights_masked.unsqueeze(-1)  # [B, N, 1]
            
            # Fuse local and global features
            fused_pixels = updated_pixels + fusion_weights_expanded * cluster_global_context_expanded
            
            # Accumulate contributions from all clusters
            final_pixel_features = final_pixel_features + (fused_pixels * cluster_mask.unsqueeze(-1))
        
        return final_pixel_features

class ModalityBranch(nn.Module):
    def __init__(self, input_channels, feature_dim, num_clusters):
        super().__init__()
        self.input_proj = nn.Conv2d(input_channels, feature_dim, 1)
        self.bn_input = nn.BatchNorm2d(feature_dim)  # ← THIS IS CORRECT (2D data)
        
        mamba_config = {
            'd_state': 16,
            'd_conv': 4,
            'expand': 2
        }
        
        self.cluster_layer = ClusterMambaLayer(feature_dim, num_clusters, mamba_config)
        
    def forward(self, x):
        # x: [B, C, H, W]
        batch_size, channels, height, width = x.shape
        
        # Project to feature dimension
        x_proj = self.input_proj(x)  # [B, feature_dim, H, W]
        x_proj = self.bn_input(x_proj)  # ← THIS IS CORRECT
        x_proj = F.gelu(x_proj)

        # Flatten spatial dimensions
        x_flat = x_proj.view(batch_size, -1, height*width).transpose(1, 2)  # [B, H*W, feature_dim]
        
        # Apply cluster-based processing
        x_processed = self.cluster_layer(x_flat, height, width)
        
        # Reshape back to spatial
        x_out = x_processed.transpose(1, 2).view(batch_size, -1, height, width)
        
        return x_out

    
###########   Check Part   ###########
""""
This part is for testing the different part of the codes

"""
import warnings
warnings.filterwarnings("ignore", message="Expected u.is_cuda() to be true")

def debug_mamba_device():
    """Check if Mamba is actually using GPU"""
    print("\n=== Debugging Mamba Device ===")
    
    device = torch.device('cuda')
    mamba = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).to(device)
    
    # Check model parameters device
    for name, param in mamba.named_parameters():
        print(f"Parameter {name}: {param.device}")
    
    # Test forward pass with detailed device checking
    test_input = torch.randn(2, 10, 64).to(device)
    print(f"Input device: {test_input.device}")
    
    # Manually check some internal states
    try:
        output = mamba(test_input)
        print(f"Output device: {output.device}")
        print(f"Output shape: {output.shape}")
        print("✅ Mamba forward pass completed successfully on GPU!")
    except Exception as e:
        print(f"❌ Mamba failed: {e}")

def setup_device():
    """Setup device and check Mamba availability"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("❌ Using CPU - Mamba may be slow or have issues")
    
    # Check if Mamba supports CPU
    try:
        # Test if Mamba works on CPU with a small tensor
        test_mamba = Mamba(d_model=64, d_state=16, d_conv=4, expand=2)
        test_input = torch.randn(1, 10, 64)
        test_output = test_mamba(test_input)
        print("✅ Mamba works on current device")
    except Exception as e:
        print(f"❌ Mamba error: {e}")
        if device.type == 'cpu':
            print("💡 Try running on GPU for better Mamba performance")
    
    return device

def test_individual_components(device):
    """Test each component individually to verify output shapes"""
    print("\n=== Testing Individual Components ===")
    
    # Test parameters
    batch_size = 2
    height, width = 13, 13
    num_pixels = height * width
    feature_dim = 64
    num_clusters = 8
    
    # Test DifferentiableHardKMeans
    print("\n1. Testing DifferentiableHardKMeans:")
    kmeans = DifferentiableHardKMeans(num_clusters, feature_dim).to(device)
    dummy_features = torch.randn(batch_size, num_pixels, feature_dim).to(device)
    cluster_assignments = kmeans(dummy_features)
    print(f"Input: {dummy_features.shape}")
    print(f"Output: {cluster_assignments.shape}")
    print(f"Cluster assignments sum per pixel: {cluster_assignments[0, :5, :].sum(dim=1)}")  # First 5 pixels
    
    # Test ClusterProcessor
    print("\n2. Testing ClusterProcessor:")
    mamba_config = {'d_state': 16, 'd_conv': 4, 'expand': 2}
    cluster_processor = ClusterProcessor(feature_dim, mamba_config).to(device)
    cluster_pixels = torch.randn(batch_size, 10, feature_dim).to(device)  # 10 pixels in a cluster
    updated_pixels = cluster_processor(cluster_pixels)
    print(f"Input: {cluster_pixels.shape}")
    print(f"Output: {updated_pixels.shape}")
    
    # Test RepresentativeCalculator
    print("\n3. Testing RepresentativeCalculator:")
    rep_calculator = RepresentativeCalculator(feature_dim).to(device)
    representative = rep_calculator(cluster_pixels)
    print(f"Input: {cluster_pixels.shape}")
    print(f"Output: {representative.shape}")
    
    # Test ClusterMambaLayer
    print("\n4. Testing ClusterMambaLayer:")
    cluster_layer = ClusterMambaLayer(feature_dim, num_clusters, mamba_config).to(device)
    all_pixels = torch.randn(batch_size, num_pixels, feature_dim).to(device)
    output = cluster_layer(all_pixels, height, width)
    print(f"Input: {all_pixels.shape}")
    print(f"Output: {output.shape}")
    
    # Test ModalityBranch
    print("\n5. Testing ModalityBranch:")
    # Test HSI branch
    hsi_branch = ModalityBranch(input_channels=138, feature_dim=feature_dim, num_clusters=8).to(device)
    hsi_input = torch.randn(batch_size, 138, height, width).to(device)
    hsi_output = hsi_branch(hsi_input)
    print(f"HSI Input: {hsi_input.shape}")
    print(f"HSI Output: {hsi_output.shape}")
    
    # Test LiDAR branch
    lidar_branch = ModalityBranch(input_channels=1, feature_dim=feature_dim, num_clusters=6).to(device)
    lidar_input = torch.randn(batch_size, 1, height, width).to(device)
    lidar_output = lidar_branch(lidar_input)
    print(f"LiDAR Input: {lidar_input.shape}")
    print(f"LiDAR Output: {lidar_output.shape}")

def test_complete_flow(device):
    """Test the complete forward pass with both modalities"""
    print("\n=== Testing Complete Flow ===")
    
    # Parameters
    batch_size = 2
    height, width = 13, 13
    feature_dim = 64
    
    # Create both branches
    hsi_branch = ModalityBranch(input_channels=138, feature_dim=feature_dim, num_clusters=8).to(device)
    lidar_branch = ModalityBranch(input_channels=1, feature_dim=feature_dim, num_clusters=6).to(device)
    
    # Create dummy data
    hsi_data = torch.randn(batch_size, 138, height, width).to(device)
    lidar_data = torch.randn(batch_size, 1, height, width).to(device)
    
    print(f"HSI data shape: {hsi_data.shape}")
    print(f"LiDAR data shape: {lidar_data.shape}")
    
    # Forward pass through both branches
    with torch.no_grad():
        hsi_features = hsi_branch(hsi_data)
        lidar_features = lidar_branch(lidar_data)
    
    print(f"HSI features shape: {hsi_features.shape}")
    print(f"LiDAR features shape: {lidar_features.shape}")
    
    # Simple fusion (concatenation)
    fused_features = torch.cat([hsi_features, lidar_features], dim=1)
    print(f"Fused features shape: {fused_features.shape}")
    
    return hsi_features, lidar_features, fused_features

def test_memory_usage(device):
    """Test memory usage and performance"""
    print("\n=== Testing Memory Usage ===")
    
    batch_size = 4
    height, width = 13, 13
    feature_dim = 64
    
    # Create model
    hsi_branch = ModalityBranch(input_channels=138, feature_dim=feature_dim, num_clusters=8).to(device)
    lidar_branch = ModalityBranch(input_channels=1, feature_dim=feature_dim, num_clusters=6).to(device)
    
    print(f"Using device: {device}")
    
    # Create data
    hsi_data = torch.randn(batch_size, 138, height, width).to(device)
    lidar_data = torch.randn(batch_size, 1, height, width).to(device)
    
    # Memory before forward pass
    if device.type == 'cuda':
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
    
    # Forward pass
    with torch.no_grad():
        hsi_features = hsi_branch(hsi_data)
        lidar_features = lidar_branch(lidar_data)
    
    # Memory after forward pass
    if device.type == 'cuda':
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_used = memory_after - memory_before
        print(f"GPU memory used: {memory_used:.2f} MB")
    
    print(f"HSI features on {device}: {hsi_features.shape}")
    print(f"LiDAR features on {device}: {lidar_features.shape}")

def count_parameters(model, model_name):
    """Count the number of parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} - Total params: {total_params:,}, Trainable: {trainable_params:,}")

def test_cpu_fallback():
    """Test if we can create a CPU-compatible version if Mamba fails"""
    print("\n=== Testing CPU Fallback ===")
    
    # If Mamba fails, we can use a simple LSTM or Linear fallback
    class CPUClusterProcessor(nn.Module):
        def __init__(self, feature_dim):
            super().__init__()
            self.feature_dim = feature_dim
            # Use LSTM instead of Mamba for CPU compatibility
            self.lstm = nn.LSTM(feature_dim, feature_dim, batch_first=True, bidirectional=True)
            self.proj = nn.Linear(feature_dim * 2, feature_dim)  # Project back to feature_dim
            self.norm = nn.LayerNorm(feature_dim)
            
        def forward(self, cluster_pixels):
            lstm_out, _ = self.lstm(cluster_pixels)
            updated_pixels = self.proj(lstm_out)
            updated_pixels = self.norm(updated_pixels)
            return updated_pixels
    
    # Test the CPU version
    processor = CPUClusterProcessor(64)
    test_input = torch.randn(2, 10, 64)
    test_output = processor(test_input)
    print(f"CPU Fallback - Input: {test_input.shape}, Output: {test_output.shape}")

def main():
    """Main function to run all tests"""
    print("Cluster Mamba Architecture Test")
    print("=" * 50)
    
    # Setup device
    device = setup_device()
    
    debug_mamba_device()

    try:
        # Test individual components
        test_individual_components(device)
        
        # Test complete flow
        hsi_features, lidar_features, fused_features = test_complete_flow(device)
        
        # Test memory usage if on GPU
        if device.type == 'cuda':
            test_memory_usage(device)
        
        # Count parameters
        print("\n=== Parameter Count ===")
        hsi_branch = ModalityBranch(input_channels=138, feature_dim=64, num_clusters=8)
        lidar_branch = ModalityBranch(input_channels=1, feature_dim=64, num_clusters=6)
        
        count_parameters(hsi_branch, "HSI Branch")
        count_parameters(lidar_branch, "LiDAR Branch")
        
        # Test with different sizes
        print("\n=== Testing Different Sizes ===")
        sizes = [(13, 13), (25, 25)]
        
        for h, w in sizes:
            print(f"\nTesting size {h}x{w}:")
            hsi_data = torch.randn(2, 138, h, w).to(device)
            lidar_data = torch.randn(2, 1, h, w).to(device)
            
            hsi_branch = ModalityBranch(input_channels=138, feature_dim=64, num_clusters=8).to(device)
            lidar_branch = ModalityBranch(input_channels=1, feature_dim=64, num_clusters=6).to(device)
            
            with torch.no_grad():
                hsi_out = hsi_branch(hsi_data)
                lidar_out = lidar_branch(lidar_data)
            
            print(f"  Input: HSI{list(hsi_data.shape)}, LiDAR{list(lidar_data.shape)}")
            print(f"  Output: HSI{list(hsi_out.shape)}, LiDAR{list(lidar_out.shape)}")
            
    except RuntimeError as e:
        print(f"\n❌ Error during testing: {e}")
        print("💡 Trying CPU fallback...")
        test_cpu_fallback()

if __name__ == "__main__":
    main()