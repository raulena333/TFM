import torch
import convNd
from typing import Tuple, Callable

class BatchNorm4d(torch.nn.Module):
    def __init__(self, channel_size):
        super(BatchNorm4d, self).__init__()
        # This approach uses BatchNorm1d by flattening the spatial dimensions.
        # It's technically correct for computing mean/variance per channel across all spatial locations
        # and batch samples, but a custom BatchNorm4d might be more direct if implemented.
        self.norm = torch.nn.BatchNorm1d(channel_size)

    def forward(self, x):
        # x shape: (N, C, D1, D2, D3, D4)
        shape_x = x.shape
        
        # Reshape to (N, C, D1*D2*D3*D4) for BatchNorm1d
        # This flattens the 4 spatial dimensions into one 'length' dimension
        x_reshaped = x.view(shape_x[0], shape_x[1], -1) 
        
        # Apply BatchNorm1d
        out_normalized = self.norm(x_reshaped)
        
        # Reshape back to original 4D shape
        out = out_normalized.view(shape_x)
        return out
    
# Function to get default kernel initializer for Conv4d/ConvTranspose4d
def _default_kernel_initializer(x_tensor):
    # Use Kaiming (He) uniform initialization, suitable for ReLU activations
    # 'fan_in' mode is common for forward pass
    torch.nn.init.kaiming_uniform_(x_tensor, nonlinearity='relu')

# Function to get default bias initializer
def _default_bias_initializer(x_tensor):
    # Initialize biases to zero
    torch.nn.init.zeros_(x_tensor)

def Conv4d(in_channels: int, out_channels: int, kernel_size:int=2, 
           stride:int=1, padding:int = 0, padding_mode: str ="zeros",   
           bias: bool = True, groups: int = 1, dilation: int = 1):
    
    # Ensure kernel_size, stride, padding, dilation are always tuples
    # This aligns with how convNd expects them and prevents potential issues
    if not isinstance(kernel_size, Tuple):
        kernel_size = tuple(kernel_size for _ in range(4)) # Assuming 4 dims for Conv4d
    if not isinstance(stride, Tuple):
        stride = tuple(stride for _ in range(4))
    if not isinstance(padding, Tuple):
        padding = tuple(padding for _ in range(4))
    if not isinstance(dilation, Tuple):
        dilation = tuple(dilation for _ in range(4))

    return convNd.convNd(in_channels=in_channels, out_channels=out_channels,
                         num_dims=4, kernel_size=kernel_size, 
                         stride=stride, padding=padding, 
                         padding_mode=padding_mode, output_padding=0, # output_padding is 0 for Conv4d
                         is_transposed=False, use_bias=bias, groups=groups, dilation = dilation, 
                         kernel_initializer=_default_kernel_initializer, # Use standard initializer
                         bias_initializer=_default_bias_initializer if bias else None) # Use standard initializer if bias is used

def ConvTranspose4d(in_channels: int, out_channels: int, kernel_size:int=2,
                    stride:int=1, padding:int = 0, padding_mode: str ="zeros", 
                    bias: bool = True, groups: int = 1, dilation: int = 1):
    
    # Ensure kernel_size, stride, padding, dilation are always tuples
    if not isinstance(kernel_size, Tuple):
        kernel_size = tuple(kernel_size for _ in range(4))
    if not isinstance(stride, Tuple):
        stride = tuple(stride for _ in range(4))
    if not isinstance(padding, Tuple):
        padding = tuple(padding for _ in range(4))
    if not isinstance(dilation, Tuple):
        dilation = tuple(dilation for _ in range(4))
    
    return convNd.convNd(in_channels=in_channels, out_channels=out_channels,
                         num_dims=4, kernel_size=kernel_size, 
                         stride=stride, padding=padding, 
                         padding_mode=padding_mode, output_padding=0, # Assume output_padding handled by convNd's logic
                         is_transposed=True, use_bias=bias, groups=groups, dilation = dilation, 
                         kernel_initializer=_default_kernel_initializer, # Use standard initializer
                         bias_initializer=_default_bias_initializer if bias else None) # Use standard initializer if bias is used