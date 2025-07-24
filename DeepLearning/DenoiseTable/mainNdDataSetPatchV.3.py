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
from convNd import convNd  # Assuming convNd is available and handles 4D
import time  # For timing the training loop
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader  # Import Dataset and DataLoader

# Set device for PyTorch operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f'Python version: {sys.version}')
print(f'Pytorch version: {torch.__version__}')
torch.backends.cudnn.deterministic = True

# --- REVISED: DummyNoise2NoisePatchDataset to generate patches from a pre-generated full volume ---
class DummyNoise2NoisePatchDataset(Dataset):
    def __init__(self, num_samples: int,
                 full_data_dims: Tuple[int, ...],  # The dimensions of the "full" data, for context
                 patch_size: Tuple[int, ...],  # Dimensions of the patches to be extracted
                 std_noise: float = 0.15):

        self.full_data_dims = full_data_dims
        self.patch_size = patch_size
        self.std_noise = std_noise
        self.num_samples = num_samples  # Number of patches to generate *from the full volume*

        # Ensure patch_size matches the number of dimensions in full_data_dims
        if len(self.patch_size) != len(self.full_data_dims):
            raise ValueError("patch_size must have the same number of dimensions as full_data_dims")

        # Ensure patch_size is not larger than full_data_dims in any dimension
        for i in range(len(self.full_data_dims)):
            if self.patch_size[i] > self.full_data_dims[i]:
                raise ValueError(f"Patch size dimension {i} ({self.patch_size[i]}) cannot be larger than full data dimension {i} ({self.full_data_dims[i]})")

        print(f"Generating full clean 4D volume of size: {self.full_data_dims}")
        # 1. Generate a single, large, "clean" 4D volume (all ones for simplicity)
        # In a real scenario, this would be loading your actual large data.
        self.clean_full_volume = torch.ones(self.full_data_dims)

        print(f"Generating two noisy versions of the full 4D volume (std_noise={self.std_noise})...")
        # 2. Generate two independent, full-sized noisy versions of this clean volume
        # We pre-generate these to simulate loading them or having them ready.
        # This makes __getitem__ faster as it only does patching.
        self.noisy_full_volume_A = self.clean_full_volume + self.std_noise * torch.randn(self.full_data_dims)
        self.noisy_full_volume_B = self.clean_full_volume + self.std_noise * torch.randn(self.full_data_dims)
        print("Full noisy volumes generated.")

    def __len__(self):
        # The number of samples is now the number of patches we want to extract
        # throughout an epoch.
        return self.num_samples

    def __getitem__(self, idx: int):
        # 3. From these two full-sized noisy volumes, extract random patches of patch_size

        # Determine random starting coordinates for the patch
        # The range for each dimension is [0, full_dim - patch_dim]
        start_coords = [
            torch.randint(0, self.full_data_dims[d] - self.patch_size[d] + 1, (1,)).item()
            for d in range(len(self.full_data_dims))
        ]

        # Extract slices for noisy_input_A and noisy_input_B
        # This uses Python's slice notation to extract the patch
        # e.g., noisy_full_volume_A[start_d0:start_d0+patch_d0, start_d1:start_d1+patch_d1, ...]
        slices = tuple(slice(s, s + p) for s, p in zip(start_coords, self.patch_size))

        input_patch = self.noisy_full_volume_A[slices]
        target_patch = self.noisy_full_volume_B[slices]

        # Add the channel dimension (assuming 1 channel)
        # Shape becomes (1, D1, D2, D3, D4) for the patch
        input_patch_with_channel = input_patch.unsqueeze(0)
        target_patch_with_channel = target_patch.unsqueeze(0)

        return input_patch_with_channel, target_patch_with_channel


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

        x = self.conv4(x)  # Output layer usually doesn't have an activation if it's regressing values
        return x

def plot_multiple_d3d4_slices(input_vol: torch.Tensor, true_clean_vol: torch.Tensor, output_vol: torch.Tensor,
                               full_data_dims: Tuple[int, ...], d1_indices: list, d2_indices: list, output_dir: str = '.'):
    """
    Plots D3-D4 slices of the 4D volumes at specified D1 and D2 indices.

    Args:
        input_vol (torch.Tensor): The noisy input 4D volume (B=1, C=1, D1, D2, D3, D4).
        true_clean_vol (torch.Tensor): The true clean 4D volume (B=1, C=1, D1, D2, D3, D4).
        output_vol (torch.Tensor): The denoised output 4D volume (B=1, C=1, D1, D2, D3, D4).
        full_data_dims (Tuple[int, ...]): Original full data dimensions (D1, D2, D3, D4).
        d1_indices (list): List of indices along the D1 dimension to plot.
        d2_indices (list): List of indices along the D2 dimension to plot.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nPlotting multiple D3-D4 slices in '{output_dir}'...")

    for d1_idx in d1_indices:
        if d1_idx >= full_data_dims[0] or d1_idx < 0:
            print(f"Warning: D1 index {d1_idx} is out of bounds for dimension size {full_data_dims[0]}. Skipping.")
            continue
        for d2_idx in d2_indices:
            if d2_idx >= full_data_dims[1] or d2_idx < 0:
                print(f"Warning: D2 index {d2_idx} is out of bounds for dimension size {full_data_dims[1]}. Skipping.")
                continue

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'4D Denoising - D3-D4 Slice at D1={d1_idx}, D2={d2_idx}')

            # Extract 2D slices for plotting (fixed D1 and D2, varying D3 and D4)
            input_slice = input_vol[0, 0, d1_idx, d2_idx, :, :].cpu().numpy()
            true_clean_slice = true_clean_vol[0, 0, d1_idx, d2_idx, :, :].cpu().numpy()
            output_slice = output_vol[0, 0, d1_idx, d2_idx, :, :].cpu().numpy()

            # Determine global vmin/vmax for consistent color scaling across plots
            all_slice_values = np.concatenate([input_slice.flatten(), true_clean_slice.flatten(), output_slice.flatten()])
            global_vmin = all_slice_values.min()
            global_vmax = all_slice_values.max()
            # Add a small buffer for better visualization
            buffer = (global_vmax - global_vmin) * 0.05
            global_vmin -= buffer
            global_vmax += buffer

            im0 = axes[0].imshow(input_slice, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
            axes[0].set_title('Noisy Input')
            fig.colorbar(im0, ax=axes[0])

            im1 = axes[1].imshow(true_clean_slice, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
            axes[1].set_title('True Clean Signal (All Ones)')
            fig.colorbar(im1, ax=axes[1])

            im2 = axes[2].imshow(output_slice, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
            axes[2].set_title('Denoised Output')
            fig.colorbar(im2, ax=axes[2])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            filename = os.path.join(output_dir, f'4D_denoising_D3D4_slice_D1_{d1_idx}_D2_{d2_idx}.png')
            plt.savefig(filename)
            plt.close(fig) # Close the figure to free up memory
            print(f"Saved: {filename}")


def test_4d_net(num_epochs: int = 5, learning_rate: float = 0.0001, batch_size: int = 4,
                full_data_dims: Tuple[int, ...] = (64, 64, 64, 64),  # Larger conceptual dimensions
                patch_size: Tuple[int, ...] = (16, 16, 16, 16),  # Size of patches
                num_patches_per_epoch: int = 500  # How many patches to train on per epoch
                ):
    """
    Tests the 4D convolutional neural network with dummy data using Noise2Noise and patching.
    """
    print(f"\n--- Testing 4D Network on {device} with Noise2Noise and Patching ---")
    print(f"Conceptual Full Data Dimensions: {full_data_dims}")
    print(f"Patch Size for Training: {patch_size}")

    kernel_size = 3

    # --- UPDATED: Create Noise2Noise Patch Dataset and DataLoader ---
    # Now, the dataset will internally generate the full noisy volumes once,
    # and then provide random patches from them during iteration.
    dataset = DummyNoise2NoisePatchDataset(num_patches_per_epoch,
                                           full_data_dims,
                                           patch_size,
                                           std_noise=0.15)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Total patches (samples) per epoch: {num_patches_per_epoch}")
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
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)  # Target is another noisy version (Noise2Noise)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(
                {
                    "Batch Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}"
                },
                refresh=True
            )

        avg_epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")

    print("Training finished.")

    # --- Evaluation and Plotting ---
    model.eval()
    with torch.no_grad():
        evaluation_input = dataset.noisy_full_volume_A.unsqueeze(0).unsqueeze(0).to(device) # (B=1, C=1, D1, D2, D3, D4)
        true_clean_signal = dataset.clean_full_volume.unsqueeze(0).unsqueeze(0).to(device) # For plotting comparison

        print(f"\nPerforming inference on a full volume (shape {evaluation_input.shape}). This might take a moment...")
        start_time = time.time()
        evaluation_output = model(evaluation_input)
        end_time = time.time()

        print(f"Output shape: {evaluation_output.shape}")
        print(f"Inference time for full volume: {(end_time - start_time):.2f} seconds")

        # --- Plotting multiple D3-D4 Slices ---
        # Choose a few representative indices for D1 and D2
        d1_middle_idx = full_data_dims[0] // 2
        d2_middle_idx = full_data_dims[1] // 2
        d1_quarter_idx = full_data_dims[0] // 4
        d2_quarter_idx = full_data_dims[1] // 4
        d1_three_quarter_idx = (full_data_dims[0] * 3) // 4
        d2_three_quarter_idx = (full_data_dims[1] * 3) // 4

        # Ensure indices are within bounds and non-negative
        selected_d1_indices = sorted(list(set([
            0, # First slice
            d1_quarter_idx,
            d1_middle_idx,
            d1_three_quarter_idx,
            full_data_dims[0] - 1 # Last slice
        ])))
        selected_d1_indices = [idx for idx in selected_d1_indices if 0 <= idx < full_data_dims[0]]


        selected_d2_indices = sorted(list(set([
            0, # First slice
            d2_quarter_idx,
            d2_middle_idx,
            d2_three_quarter_idx,
            full_data_dims[1] - 1 # Last slice
        ])))
        selected_d2_indices = [idx for idx in selected_d2_indices if 0 <= idx < full_data_dims[1]]

        plot_multiple_d3d4_slices(evaluation_input, true_clean_signal, evaluation_output,
                                  full_data_dims, selected_d1_indices, selected_d2_indices, output_dir='denoising_d3d4_slices')


        # --- Plotting a single D1-D2 slice (e.g., central spatial slice) for context ---
        # This is the original D1-D2 plotting, kept for completeness, renamed.
        d3_slice_idx = full_data_dims[2] // 2
        d4_slice_idx = full_data_dims[3] // 2

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'4D Denoising - D1-D2 Slice at D3={d3_slice_idx}, D4={d4_slice_idx}')

        input_slice_d1d2 = evaluation_input[0, 0, :, :, d3_slice_idx, d4_slice_idx].cpu().numpy()
        true_clean_slice_d1d2 = true_clean_signal[0, 0, :, :, d3_slice_idx, d4_slice_idx].cpu().numpy()
        output_slice_d1d2 = evaluation_output[0, 0, :, :, d3_slice_idx, d4_slice_idx].cpu().numpy()

        all_slice_values_d1d2 = np.concatenate([input_slice_d1d2.flatten(), true_clean_slice_d1d2.flatten(), output_slice_d1d2.flatten()])
        global_vmin_d1d2 = all_slice_values_d1d2.min()
        global_vmax_d1d2 = all_slice_values_d1d2.max()
        buffer_d1d2 = (global_vmax_d1d2 - global_vmin_d1d2) * 0.05
        global_vmin_d1d2 -= buffer_d1d2
        global_vmax_d1d2 += buffer_d1d2

        im0 = axes[0].imshow(input_slice_d1d2, cmap='viridis', vmin=global_vmin_d1d2, vmax=global_vmax_d1d2)
        axes[0].set_title('Noisy Input (D1-D2 Slice)')
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(true_clean_slice_d1d2, cmap='viridis', vmin=global_vmin_d1d2, vmax=global_vmax_d1d2)
        axes[1].set_title('True Clean Signal (D1-D2 Slice)')
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(output_slice_d1d2, cmap='viridis', vmin=global_vmin_d1d2, vmax=global_vmax_d1d2)
        axes[2].set_title('Denoised Output (D1-D2 Slice)')
        fig.colorbar(im2, ax=axes[2])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('4D_denoising_central_d1d2_slice.png')
        plt.close()


        # --- Histogram Plotting ---
        input_flat = evaluation_input.cpu().numpy().flatten()
        true_clean_flat = true_clean_signal.cpu().numpy().flatten()
        output_flat = evaluation_output.cpu().numpy().flatten()

        plt.figure(figsize=(10, 6))
        
        # Determine global min/max for histogram bins based on all data
        all_data_flat = np.concatenate([input_flat, true_clean_flat, output_flat])
        hist_vmin = all_data_flat.min()
        hist_vmax = all_data_flat.max()
        hist_buffer = (hist_vmax - hist_vmin) * 0.05
        bins = np.linspace(hist_vmin - hist_buffer, hist_vmax + hist_buffer, 50)


        plt.hist(input_flat, bins=bins, alpha=0.5, label='Noisy Input Histogram', color='red', density=True)
        plt.hist(true_clean_flat, bins=bins, alpha=0.5, label='True Clean Histogram (All Ones)', color='green', density=True)
        plt.hist(output_flat, bins=bins, alpha=0.5, label='Denoised Output Histogram', color='blue', density=True)

        plt.title('Histogram of Pixel Intensities (Full Volume)')
        plt.xlabel('Pixel Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('4D_denoising_histogram_full_volume.png')
        plt.close()


if __name__ == "__main__":
    test_4d_net(num_epochs=5,
                learning_rate=0.0005,
                batch_size=4,
                full_data_dims=(50, 50, 100, 100), # D1, D2, D3, D4
                patch_size=(16, 16, 16, 16),
                num_patches_per_epoch=500
               )