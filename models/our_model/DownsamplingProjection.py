import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsamplingProjection(nn.Module):
    def __init__(self, in_channels, out_channels=None, method='conv', kernel_size=3, stride=2, padding=1):
        """
        Downsampling projection to reduce spatial size by factor 2
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (default: same as input)
            method: Downsampling method - 'conv', 'pool', 'pixel_shuffle', 'linear'
            kernel_size: Kernel size for conv method
            stride: Stride for conv method
            padding: Padding for conv method
        """
        super(DownsamplingProjection, self).__init__()
        
        self.method = method
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        
        if method == 'conv':
            # Convolutional downsampling
            self.downsample = nn.Conv2d(
                in_channels, self.out_channels, 
                kernel_size=kernel_size, stride=stride, 
                padding=padding, bias=False
            )
            self.norm = nn.BatchNorm2d(self.out_channels)
            self.activation = nn.ReLU(inplace=True)
            
        elif method == 'pool':
            # Max pooling + projection
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            if in_channels != self.out_channels:
                self.proj = nn.Conv2d(in_channels, self.out_channels, 1, bias=False)
                self.norm = nn.BatchNorm2d(self.out_channels)
            else:
                self.proj = nn.Identity()
                self.norm = nn.Identity()
            self.activation = nn.ReLU(inplace=True)
            
        elif method == 'pixel_shuffle':
            # Pixel shuffle (inverse) for downsampling
            self.shuffle = nn.PixelUnshuffle(2)  # This increases channels by 4x
            # After unshuffle: [B, C*4, H/2, W/2], so we need to project back
            self.proj = nn.Conv2d(in_channels * 4, self.out_channels, 1, bias=False)
            self.norm = nn.BatchNorm2d(self.out_channels)
            self.activation = nn.ReLU(inplace=True)
            
        elif method == 'linear':
            # Linear projection for patches (Vision Transformer style)
            self.proj = nn.Conv2d(
                in_channels, self.out_channels, 
                kernel_size=2, stride=2, bias=False
            )
            self.norm = nn.BatchNorm2d(self.out_channels)
            self.activation = nn.ReLU(inplace=True)
            
        else:
            raise ValueError(f"Unknown downsampling method: {method}")

    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
        Returns:
            Output feature map [B, C_out, H/2, W/2]
        """
        if self.method == 'conv':
            x = self.downsample(x)
            x = self.norm(x)
            x = self.activation(x)
            
        elif self.method == 'pool':
            x = self.pool(x)
            x = self.proj(x)
            x = self.norm(x)
            x = self.activation(x)
            
        elif self.method == 'pixel_shuffle':
            x = self.shuffle(x)  # [B, C*4, H/2, W/2]
            x = self.proj(x)     # [B, C_out, H/2, W/2]
            x = self.norm(x)
            x = self.activation(x)
            
        elif self.method == 'linear':
            x = self.proj(x)  # [B, C_out, H/2, W/2]
            x = self.norm(x)
            x = self.activation(x)
            
        return x

    def extra_repr(self):
        return f'method={self.method}, in_channels={self.in_channels}, out_channels={self.out_channels}'


class OptionalDownsampling(nn.Module):
    def __init__(self, in_channels, out_channels=None, method='conv', enabled=True, **kwargs):
        """
        Optional downsampling module that can be disabled
        
        Args:
            enabled: If False, becomes identity mapping
            Other args same as DownsamplingProjection
        """
        super(OptionalDownsampling, self).__init__()
        self.enabled = enabled
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        
        if enabled:
            self.downsample = DownsamplingProjection(
                in_channels, out_channels, method, **kwargs
            )
        else:
            # When disabled, we need to ensure the output has the right channels
            if in_channels != self.out_channels:
                self.downsample = nn.Conv2d(in_channels, self.out_channels, 1, bias=False)
            else:
                self.downsample = nn.Identity()

    def forward(self, x):
        return self.downsample(x)

    def set_enabled(self, enabled):
        """Dynamically enable/disable downsampling"""
        if enabled != self.enabled:
            self.enabled = enabled
            if enabled:
                # Re-initialize downsampling
                self.downsample = DownsamplingProjection(
                    self.in_channels, self.out_channels, method='conv'
                )
            else:
                # Use identity or projection only
                if self.in_channels != self.out_channels:
                    self.downsample = nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False)
                else:
                    self.downsample = nn.Identity()


class FeatureDownsampler(nn.Module):
    """
    Simple feature downsampler that can be easily integrated into any model
    """
    def __init__(self, channels, method='conv', enabled=True):
        super(FeatureDownsampler, self).__init__()
        self.enabled = enabled
        self.downsample = OptionalDownsampling(channels, channels, method=method, enabled=enabled)
        
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
        Returns:
            Downsampled feature map [B, C, H/2, W/2] if enabled, else same as input
        """
        return self.downsample(x)
    
    def set_enabled(self, enabled):
        """Enable/disable downsampling"""
        self.downsample.set_enabled(enabled)


# Test the downsampling functionality
if __name__ == "__main__":
    # Test parameters
    batch_size, channels, height, width = 4, 64, 128, 128
    
    print("Testing Downsampling Projection")
    print("=" * 50)
    
    # Test input
    input_tensor = torch.randn(batch_size, channels, height, width)
    print(f"Input shape: {input_tensor.shape}")
    
    # Test all downsampling methods
    methods = ['conv', 'pool', 'pixel_shuffle', 'linear']
    
    for method in methods:
        try:
            downsample = DownsamplingProjection(channels, method=method)
            output = downsample(input_tensor)
            print(f"{method:15s}: {input_tensor.shape} -> {output.shape}")
            
        except Exception as e:
            print(f"{method:15s}: Failed - {e}")
    
    # Test optional downsampling
    print("\nTesting Optional Downsampling:")
    optional_ds = OptionalDownsampling(channels, enabled=True)
    output_enabled = optional_ds(input_tensor)
    print(f"Enabled:  {input_tensor.shape} -> {output_enabled.shape}")
    
    optional_ds.set_enabled(False)
    output_disabled = optional_ds(input_tensor)
    print(f"Disabled: {input_tensor.shape} -> {output_disabled.shape}")
    
    # Test FeatureDownsampler
    print("\nTesting FeatureDownsampler:")
    feature_ds = FeatureDownsampler(channels, enabled=True)
    output_feature = feature_ds(input_tensor)
    print(f"FeatureDownsampler (enabled):  {input_tensor.shape} -> {output_feature.shape}")
    
    feature_ds.set_enabled(False)
    output_feature_disabled = feature_ds(input_tensor)
    print(f"FeatureDownsampler (disabled): {input_tensor.shape} -> {output_feature_disabled.shape}")
    
    # Verify spatial reduction
    print(f"\nSpatial reduction when enabled: {height}x{width} -> {output_enabled.shape[2]}x{output_enabled.shape[3]}")
    print(f"Reduction factor: {height/output_enabled.shape[2]:.1f}x")
    
    # Test with different input sizes
    print("\nTesting different input sizes:")
    test_sizes = [(64, 64), (128, 128), (256, 256), (13, 13), (17, 17)]
    
    for h, w in test_sizes:
        test_input = torch.randn(2, channels, h, w)
        downsample = DownsamplingProjection(channels, method='conv')
        output = downsample(test_input)
        reduction_h = h / output.shape[2]
        reduction_w = w / output.shape[3]
        print(f"Input {h:3d}x{w:3d} -> Output {output.shape[2]:3d}x{output.shape[3]:3d} (reduction: {reduction_h:.1f}x{reduction_w:.1f})")
    
    # Test channel change
    print("\nTesting channel change:")
    input_tensor = torch.randn(2, 32, 64, 64)
    downsample = DownsamplingProjection(32, 64, method='conv')
    output = downsample(input_tensor)
    print(f"Channel change: {input_tensor.shape} -> {output.shape}")