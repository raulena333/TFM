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
from torch.utils.data import Dataset, DataLoader # Import Dataset and DataLoader

# Set device for PyTorch operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f'Python version: {sys.version}')
print(f'Pytorch version: {torch.__version__}')
torch.backends.cudnn.deterministic = True

# --- NEW: Define a custom Dataset for your dummy data ---
class DummyNoiseDataset(Dataset):
    def __init__(self, num_samples: int,
                 input_dims: Tuple[int, ...],
                 output_dims: Tuple[int, ...],
                 std_noise: float = 0.15):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.std_noise = std_noise
        self.num_samples = num_samples
        
        # Pre-generate all clean "signals" if they are the same across samples
        # For "all ones" base signal, no need to store it explicitly per sample.
        # We will generate noisy versions on the fly in __getitem__.
        # If your "clean" signals were different for each sample, you would
        # pre-generate and store them here (e.g., self.clean_inputs = [...])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        # Generate a distinct noisy input and target for each call
        base_input_sample = torch.ones(self.input_dims)
        base_target_sample = torch.ones(self.output_dims) 
        
        noisy_input_sample = base_input_sample + self.std_noise * torch.randn(self.input_dims)
        noisy_target_sample = base_target_sample + self.std_noise * torch.randn(self.output_dims)
        
        # Add the channel dimension (assuming 1 channel)
        # Shape becomes (1, D1, D2, D3, D4)
        input_sample_with_channel = noisy_input_sample.unsqueeze(0)
        target_sample_with_channel = noisy_target_sample.unsqueeze(0)
        
        return input_sample_with_channel, target_sample_with_channel

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

def test_4d_net(num_epochs: int = 1, learning_rate: float = 0.0001, batch_size: int = 4): # Added batch_size parameter
    """
    Tests the 4D convolutional neural network with dummy data and plots slices.
    """
    print(f"\n--- Testing 4D Network on {device} ---")

    # Define dataset parameters
    total_num_samples = 100 # Total number of distinct samples in the dataset
    input_dims = (20, 20, 20, 20) # Example 4D input
    output_dims = (20, 20, 20, 20) # Example 4D output (same size for simplicity)
    kernel_size = 3

    # --- NEW: Create a dataset instance and a DataLoader ---
    dataset = DummyNoiseDataset(total_num_samples, input_dims, output_dims, std_noise=0.15)
    # Shuffling is important for training stability, especially with small datasets
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Total samples in dataset: {total_num_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches per epoch: {len(dataloader)}")


    # Instantiate the 4D network
    model = Conv4dNet(in_channels=1, out_channels=1, kernel_size=kernel_size).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with tqdm
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0 # To track loss over the epoch
        
        # Loop through batches provided by the DataLoader
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update tqdm's postfix with current batch loss and moving average loss
            pbar.set_postfix(
                {
                    "Batch Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}"
                }, 
                refresh=True
            )
        
        # After each epoch, you can print the average loss for the epoch
        avg_epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")

    print("Training finished.")

    # Evaluation and Plotting
    model.eval()
    with torch.no_grad():
        # Take the first sample from the dataset for visualization
        # To get a single sample, we need to get it from the dataset, not the dataloader
        sample_input, sample_target = dataset[0] # Get the first sample (input, target)
        sample_input = sample_input.unsqueeze(0).to(device) # Add batch dim and move to device
        sample_target = sample_target.unsqueeze(0).to(device) # Add batch dim and move to device


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
    test_4d_net(batch_size=4)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  