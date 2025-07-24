import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from typing import Tuple, Callable
import math
import sys 
from tqdm import tqdm
from convNd import convNd 
import time # For timing the training loop
import matplotlib.pyplot as plt

# Set device for PyTorch operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f'Python version: {sys.version}')
print(f'Pytorch version: {torch.__version__}')
torch.backends.cudnn.deterministic = True


def create_dummy_dataset(num_samples: int,
                         input_dims: Tuple[int, ...],
                         output_dims: Tuple[int, ...],
                         std_noise: float = 0.15):
    """
    Creates a dummy dataset suitable for Noise2Noise training.
    Each sample in the batch consists of:
    - An input tensor: 'all ones' with random noise (std_noise).
    - A target tensor: 'all ones' with *different*, independent random noise (std_noise).

    Args:
        num_samples (int): Number of distinct samples in the dataset (which will also be the batch size).
        input_dims (Tuple[int, ...]): Spatial dimensions of the input data (e.g., D1, D2, D3, D4).
        output_dims (Tuple[int, ...]): Spatial dimensions of the output data (e.g., D1, D2, D3, D4).
        std_noise (float): Standard deviation of Gaussian noise to be added to both input and target.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing (input_data, target_data),
                                           each of shape (num_samples, 1, *spatial_dims).
    """
    input_data_list = []
    target_data_list = []

    # Loop num_samples times to create each distinct sample
    for _ in range(num_samples):
        # 1. Create base data for a single sample (all ones)
        # These are the 'clean' underlying signals for this specific sample
        base_input_sample = torch.ones(input_dims)
        base_target_sample = torch.ones(output_dims) 
        
        # 2. Add independent random noise to *each* base sample
        # IMPORTANT: torch.randn() is called twice *for each loop iteration*,
        # ensuring the noise for input and target are uncorrelated.
        noisy_input_sample = base_input_sample + std_noise * torch.randn(input_dims)
        noisy_target_sample = base_target_sample + std_noise * torch.randn(output_dims)
        
        # 3. Add the channel dimension (assuming 1 channel)
        # Shape becomes (1, D1, D2, D3, D4) for this single sample
        input_sample_with_channel = noisy_input_sample.unsqueeze(0)
        target_sample_with_channel = noisy_target_sample.unsqueeze(0)
        
        # Add the processed single sample (with channel dim) to our lists
        input_data_list.append(input_sample_with_channel)
        target_data_list.append(target_sample_with_channel)
    
    # 4. Stack all the distinct samples to create the batch dimension
    # The result will be (num_samples, 1, D1, D2, D3, D4)
    input_data_batched = torch.stack(input_data_list, dim=0)
    target_data_batched = torch.stack(target_data_list, dim=0)
    
    print(f"Input data shape: {input_data_batched.shape}")
    print(f"Target data shape: {target_data_batched.shape}")
    
    # Move the entire batched tensor to the specified device
    return input_data_batched.to(device), target_data_batched.to(device)


# class Conv4dNet(nn.Module):
#     """
#     A simple 4D convolutional neural network for denoising.
#     """
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
#         super(Conv4dNet, self).__init__()
#         # Ensure num_dims is 4 for a 4D network
#         # kernel_size//2 provides 'same' padding
#         self.conv1 = convNd(in_channels, 16, num_dims=4, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
#         self.relu1 = nn.ReLU()
#         self.conv2 = convNd(16, out_channels, num_dims=4, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

#     def forward(self, x):
#         x = self.relu1(self.conv1(x))
#         x = self.conv2(x)
#         return x

class Conv4dNet(nn.Module):
    """
    A more complex 4D convolutional neural network for denoising.
    Increased depth and width compared to the previous version.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(Conv4dNet, self).__init__()
        
        # Helper for padding to maintain same spatial dimensions
        padding = kernel_size // 2

        # Layer 1: Increased channels
        self.conv1 = convNd(in_channels, 32, num_dims=4, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu1 = nn.ReLU()
        # self.bn1 = nn.BatchNorm4d(32) # If a BatchNorm4d were available

        # Layer 2: Further increased channels (bottleneck/feature extraction)
        self.conv2 = convNd(32, 64, num_dims=4, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu2 = nn.ReLU()
        # self.bn2 = nn.BatchNorm4d(64)

        # Layer 3: Decreased channels
        self.conv3 = convNd(64, 32, num_dims=4, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu3 = nn.ReLU()
        # self.bn3 = nn.BatchNorm4d(32)

        # Output Layer: Map back to desired output channels
        self.conv4 = convNd(32, out_channels, num_dims=4, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        # x = self.bn1(x) # Uncomment if BatchNorm4d is added
        
        x = self.relu2(self.conv2(x))
        # x = self.bn2(x)

        x = self.relu3(self.conv3(x))
        # x = self.bn3(x)

        x = self.conv4(x) # Output layer usually doesn't have an activation if it's regressing values
        return x

def test_4d_net(num_epochs: int = 250, learning_rate: float = 0.0001):
    """
    Tests the 4D convolutional neural network with dummy data and plots slices.
    """
    print(f"\n--- Testing 4D Network on {device} ---")

    # Define dataset parameters
    num_samples = 10 # Total number of samples in your dataset
    input_dims = (15, 15, 15, 15)  # Example 4D input
    output_dims = (15, 15, 15, 15) # Example 4D output (same size for simplicity)
    kernel_size = 3

    # Create dummy dataset
    input_data, target_data = create_dummy_dataset(
        num_samples, input_dims, output_dims, std_noise=0.15 # Use a fixed std_noise here
    )

    # Instantiate the 4D network
    model = Conv4dNet(in_channels=1, out_channels=1, kernel_size=kernel_size).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with tqdm
    print("Starting training...")
    # Corrected: Assign the tqdm instance to a variable (e.g., 'pbar')
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar: # Iterate over the pbar instance
        model.train()
        optimizer.zero_grad()
        
        outputs = model(input_data)
        loss = criterion(outputs, target_data)
        
        loss.backward()
        optimizer.step()
        
        # Update tqdm's postfix with current loss and samples processed
        samples_processed_this_epoch = num_samples
        total_samples_processed = (epoch + 1) * num_samples
        
        # Corrected: Call set_postfix on the 'pbar' instance
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}", 
                "Samples/Epoch": samples_processed_this_epoch,
                "Total Samples": total_samples_processed
            }, 
            refresh=True # Set refresh=True to force an immediate update of postfix
                          # You can experiment with False if you prefer less frequent updates
                          # but True is generally better for seeing dynamic values
        )

    print("Training finished.")

    # Evaluation and Plotting
    model.eval()
    with torch.no_grad():
        # Take the first sample from the dataset for visualization
        # .clone().detach() is good practice to get a copy that won't interfere with original graph
        sample_input = input_data[0:1].clone().detach() 
        sample_target = target_data[0:1].clone().detach()

        start_time = time.time()
        sample_output = model(sample_input)
        end_time = time.time()
        
        print(f"\nInference on a single sample (shape {sample_input.shape}):")
        print(f"Output shape: {sample_output.shape}")
        print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")

        # --- Plotting 2D Slices ---
        # A 4D tensor (B, C, D1, D2, D3, D4)
        # We want to plot (D1, D2) slices. This means we need to fix C, D3, and D4.
        
        # Select specific indices for D3 and D4.
        # Let's pick the middle slice for D3 and D4 for visualization.
        d3_slice_idx = input_dims[2] // 2
        d4_slice_idx = input_dims[3] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'4D Denoising - D1-D2 Slice at D3={d3_slice_idx}, D4={d4_slice_idx}')

        # Convert to numpy for plotting and remove batch/channel dimensions
        input_slice = sample_input[0, 0, :, :, d3_slice_idx, d4_slice_idx].cpu().numpy()
        target_slice = sample_target[0, 0, :, :, d3_slice_idx, d4_slice_idx].cpu().numpy()
        output_slice = sample_output[0, 0, :, :, d3_slice_idx, d4_slice_idx].cpu().numpy()

        # --- Customize Colorbar Range ---
        # Determine a common min and max value for the colorbar across all plots
        all_slice_values = np.concatenate([input_slice.flatten(), target_slice.flatten(), output_slice.flatten()])
        global_vmin = all_slice_values.min()
        global_vmax = all_slice_values.max()
        
        # Add a small buffer to the range for better visualization
        buffer = (global_vmax - global_vmin) * 0.05 
        global_vmin -= buffer
        global_vmax += buffer

        # Plot Input Slice
        im0 = axes[0].imshow(input_slice, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
        axes[0].set_title('Noisy Input')
        fig.colorbar(im0, ax=axes[0])

        # Plot Target Slice
        im1 = axes[1].imshow(target_slice, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
        axes[1].set_title('Noisy Target (True Clean is 1.0)')
        fig.colorbar(im1, ax=axes[1])

        # Plot Denoised Output Slice
        im2 = axes[2].imshow(output_slice, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
        axes[2].set_title('Denoised Output')
        fig.colorbar(im2, ax=axes[2])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.savefig('4D_denoising_slices.png')
        plt.close()

        # --- Denoise the whole histogram ---
        # This means comparing the distribution of pixel values (intensities)
        # before and after denoising against the target distribution.
        
        # Flatten all spatial dimensions into a single array for histograms
        input_flat = sample_input.cpu().numpy().flatten()
        target_flat = sample_target.cpu().numpy().flatten()
        output_flat = sample_output.cpu().numpy().flatten()

        plt.figure(figsize=(10, 6))
        # Use a common bin range for clear comparison
        # Using the same global vmin/vmax for histogram x-axis range
        bins = np.linspace(global_vmin, global_vmax, 50) # 50 bins

        plt.hist(input_flat, bins=bins, alpha=0.5, label='Noisy Input Histogram', color='red', density=True)
        plt.hist(target_flat, bins=bins, alpha=0.5, label='Noisy Target Histogram', color='green', density=True)
        plt.hist(output_flat, bins=bins, alpha=0.5, label='Denoised Output Histogram', color='blue', density=True)
        
        plt.title('Histogram of Pixel Intensities (Whole Sample)')
        plt.xlabel('Pixel Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('4D_denoising_histogram.png')
        plt.close()


if __name__ == "__main__":
    # Run the 4D network test with plotting
    test_4d_net()