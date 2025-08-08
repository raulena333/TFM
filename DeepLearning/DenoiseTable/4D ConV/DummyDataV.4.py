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
                         std_noise: float = 0.15,
                         signal_patch_ratio: float = 0.8): # New argument for signal-centered patches

        self.full_data_dims = full_data_dims
        self.patch_size = patch_size
        self.std_noise = std_noise
        self.num_samples = num_samples  # Number of patches to generate *from the full volume*
        self.signal_patch_ratio = signal_patch_ratio # Proportion of patches to center on signal

        # Ensure patch_size matches the number of dimensions in full_data_dims
        if len(self.patch_size) != len(self.full_data_dims):
            raise ValueError("patch_size must have the same number of dimensions as full_data_dims")

        # Ensure patch_size is not larger than full_data_dims in any dimension
        for i in range(len(self.full_data_dims)):
            if self.patch_size[i] > self.full_data_dims[i]:
                raise ValueError(f"Patch size dimension {i} ({self.patch_size[i]}) cannot be larger than full data dimension {i} ({self.full_data_dims[i]})")

        print(f"Generating full clean 4D volume of size: {self.full_data_dims} with complex patterns and zeros...")
        # 1. Generate a single, large, "clean" 4D volume with complex patterns and zero regions
        self.clean_full_volume = self._generate_complex_clean_volume(full_data_dims)

        # Pre-compute coordinates of non-zero voxels for signal-centered patches
        self.signal_coords = (self.clean_full_volume != 0).nonzero(as_tuple=False).tolist()
        print(f"Found {len(self.signal_coords)} non-zero (signal) voxels in the clean volume.")
        if not self.signal_coords:
            print("Warning: No signal (non-zero voxels) found in the generated clean volume. All patches will be random.")


        print(f"Generating two noisy versions of the full 4D volume (std_noise={self.std_noise}), applying noise ONLY to signal regions...")
        # 2. Generate two independent, full-sized noisy versions of this clean volume
        # Create a mask where clean_full_volume is non-zero (i.e., where there is signal)
        signal_mask = (self.clean_full_volume != 0).float()

        # Generate noise and apply it only where the signal_mask is 1
        noise_A = self.std_noise * torch.randn(self.full_data_dims) * signal_mask
        noise_B = self.std_noise * torch.randn(self.full_data_dims) * signal_mask

        self.noisy_full_volume_A = self.clean_full_volume + noise_A
        self.noisy_full_volume_B = self.clean_full_volume + noise_B
        print("Full noisy volumes generated with selective noise application.")

    def _generate_complex_clean_volume(self, dims: Tuple[int, ...]) -> torch.Tensor:
        """
        Generates a complex 4D clean volume where each D3-D4 slice has
        EXACTLY ONE patterned quadrant, and the other three quadrants are zeros.
        The pattern type and its quadrant are determined by D1 and D2 indices.
        """
        volume = torch.zeros(dims) # Initialize entire volume to zeros
        d1_dim, d2_dim, d3_dim, d4_dim = dims

        d3_half, d4_half = d3_dim // 2, d4_dim // 2

        # Define the four quadrant pattern functions (now distinct structured patterns)
        def get_tl_pattern(d3_h, d4_h):
            # Sine wave pattern for Top-Left
            return torch.sin(
                torch.linspace(0, 2 * math.pi, d3_h).unsqueeze(1) * 2 + # Increased frequency for visual clarity
                torch.linspace(0, 2 * math.pi, d4_h).unsqueeze(0) * 2
            ) * 0.4 + 0.5 # Scale to roughly [0.1, 0.9]

        def get_tr_pattern(d3_h, d4_h, d1_idx):
            # Cosine with exponential decay for Top-Right
            return torch.cos(
                torch.linspace(0, math.pi, d3_h).unsqueeze(1) * (d1_idx % 3 + 1)
            ) * torch.exp(
                -torch.linspace(0, 2, d4_h).unsqueeze(0) # Stronger decay
            ) * 0.6 + 0.2 # Scale to roughly [0.2, 0.8]

        def get_bl_pattern(d3_h, d4_h):
            # Radial pattern for Bottom-Left
            y_coords = torch.linspace(-1, 1, d3_h).unsqueeze(1)
            x_coords = torch.linspace(-1, 1, d4_h).unsqueeze(0)
            radius = torch.sqrt(y_coords**2 + x_coords**2)
            return torch.exp(-radius * 3) * 0.7 + 0.1 # Exponential decay from center, scale to [0.1, 0.8]

        def get_br_pattern(d3_h, d4_h):
            # Diagonal gradient pattern for Bottom-Right
            grid_d3, grid_d4 = torch.meshgrid(
                torch.arange(d3_h, dtype=torch.float32),
                torch.arange(d4_h, dtype=torch.float32),
                indexing='ij'
            )
            pattern = (grid_d3 / d3_h + grid_d4 / d4_h) / 2 # Normalize to [0,1]
            return pattern * 0.6 + 0.2 # Scale to [0.2, 0.8]


        for d1 in range(d1_dim): # M index
            for d2 in range(d2_dim): # E index
                # Determine which of the 4 quadrants gets a pattern based on (d1 + d2) sum
                # The other 3 quadrants will remain zero.
                pattern_quadrant_choice = (d1 + d2) % 4

                if pattern_quadrant_choice == 0:
                    # Top-left quadrant gets its specific pattern
                    volume[d1, d2, :d3_half, :d4_half] = get_tl_pattern(d3_half, d4_half)
                elif pattern_quadrant_choice == 1:
                    # Top-right quadrant gets its specific pattern
                    volume[d1, d2, :d3_half, d4_half:] = get_tr_pattern(d3_half, d4_dim - d4_half, d1)
                elif pattern_quadrant_choice == 2:
                    # Bottom-left quadrant gets its specific pattern
                    volume[d1, d2, d3_half:, :d4_half] = get_bl_pattern(d3_dim - d3_half, d4_half)
                else: # pattern_quadrant_choice == 3
                    # Bottom-right quadrant gets its specific pattern
                    volume[d1, d2, d3_half:, d4_half:] = get_br_pattern(d3_dim - d3_half, d4_dim - d4_half)

        # Ensure values are within a reasonable range (e.g., 0 to 1 for images)
        volume = torch.clamp(volume, 0.0, 1.0)
        return volume

    def __len__(self):
        # The number of samples is now the number of patches we want to extract
        # throughout an epoch.
        return self.num_samples

    def __getitem__(self, idx: int):
        # 3. From these two full-sized noisy volumes, extract random patches of patch_size

        start_coords = [0] * len(self.full_data_dims) # Initialize start_coords

        # Decide whether to center on a signal voxel or pick randomly
        if random.random() < self.signal_patch_ratio and self.signal_coords:
            # Try to center on a signal voxel
            # Pick a random signal voxel coordinate
            center_coord = random.choice(self.signal_coords)

            for d in range(len(self.full_data_dims)):
                # Calculate the potential start coordinate if centered
                potential_start = center_coord[d] - self.patch_size[d] // 2

                # Clamp to ensure the patch is within bounds
                max_start = self.full_data_dims[d] - self.patch_size[d]
                start_coords[d] = max(0, min(potential_start, max_start))
        else:
            # Pick a fully random starting coordinate
            for d in range(len(self.full_data_dims)):
                start_coords[d] = torch.randint(0, self.full_data_dims[d] - self.patch_size[d] + 1, (1,)).item()

        # --- DEBUGGING PRINT STATEMENT ---
        print(f"Patch {idx}: Start Coordinates (D1, D2, D3, D4): {start_coords}")

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
    A 4D convolutional neural network for denoising.
    Designed to be a middle ground in complexity/parameters between previous versions.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, base_channels: int = 40): # Adjusted base_channels
        super(Conv4dNet, self).__init__()

        # Helper for padding to maintain same spatial dimensions
        padding = kernel_size // 2

        # Layer 1: Start with base_channels
        self.conv1 = convNd(in_channels, base_channels, num_dims=4, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu1 = nn.ReLU(inplace=True)

        # Layer 2: Increased channels (double base_channels)
        self.conv2 = convNd(base_channels, base_channels * 2, num_dims=4, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu2 = nn.ReLU(inplace=True)

        # Layer 3: Further increased channels (quadruple base_channels)
        self.conv3 = convNd(base_channels * 2, base_channels * 4, num_dims=4, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu3 = nn.ReLU(inplace=True)

        # Layer 4: Decreased channels
        self.conv4 = convNd(base_channels * 4, base_channels * 2, num_dims=4, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu4 = nn.ReLU(inplace=True)

        # Layer 5: Decreased channels further
        self.conv5 = convNd(base_channels * 2, base_channels, num_dims=4, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu5 = nn.ReLU(inplace=True)

        # Output Layer: Map back to desired output channels
        self.conv6 = convNd(base_channels, out_channels, num_dims=4, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.conv6(x) # Output layer usually doesn't have an activation if it's regressing values
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
            axes[1].set_title('True Clean Signal')
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
                num_patches_per_epoch: int = 500,  # How many patches to train on per epoch
                base_channels: int = 40, # Adjusted base_channels for a middle-ground parameter count
                signal_patch_ratio: float = 0.8 # New parameter for the patch sampling strategy
                ):
    """
    Tests the 4D convolutional neural network with dummy data using Noise2Noise and patching.
    Uses the Conv4dNet with adjusted base_channels for a middle-ground parameter count.
    """
    print(f"\n--- Testing Middle-Ground 4D Network on {device} with Noise2Noise and Patching ---")
    print(f"Conceptual Full Data Dimensions: {full_data_dims}")
    print(f"Patch Size for Training: {patch_size}")
    print(f"Base Channels: {base_channels}")
    print(f"Signal-centered patch ratio: {signal_patch_ratio * 100:.0f}%")


    # --- UPDATED: Create Noise2Noise Patch Dataset and DataLoader ---
    # Now, the dataset will internally generate the full noisy volumes once,
    # and then provide random patches from them during iteration.
    dataset = DummyNoise2NoisePatchDataset(num_patches_per_epoch,
                                           full_data_dims,
                                           patch_size,
                                           std_noise=0.15,
                                           signal_patch_ratio=signal_patch_ratio) # Pass the new ratio
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Total patches (samples) per epoch: {num_patches_per_epoch}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches per epoch: {len(dataloader)}")

    kernel_size = 3 # Kept consistent

    # Instantiate the 4D network with adjusted channels
    model = Conv4dNet(in_channels=1, out_channels=1, kernel_size=kernel_size,
                      base_channels=base_channels).to(device)

    # Print model summary to see the increased parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} trainable parameters.")


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
                                  full_data_dims, selected_d1_indices, selected_d2_indices, output_dir='denoising_d3d4_slices_signal_only_noise')

if __name__ == "__main__":
    test_4d_net(num_epochs=1,
                learning_rate=0.0005,
                batch_size=4,
                full_data_dims=(50, 50, 100, 100), # D1, D2, D3, D4
                patch_size=(16, 16, 16, 16),
                num_patches_per_epoch=500, # Keeping it at 500, but feel free to adjust
                base_channels=20, # This is the key change for parameter count
                signal_patch_ratio=0.8 # 80% signal-centered, 20% random
               )