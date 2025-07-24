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

# --- REVISED: RealNoise2NoisePatchDataset to load data from NPZ ---
class RealNoise2NoisePatchDataset(Dataset):
    def __init__(self, npz_filepath: str,
                 patch_size: Tuple[int, ...],  # Dimensions of the patches to be extracted
                 num_samples: int,             # Number of patches to generate per epoch
                 signal_patch_ratio: float = 0.8): # Proportion of patches to center on signal

        if not os.path.exists(npz_filepath):
            raise FileNotFoundError(f"NPZ file not found at: {npz_filepath}")

        print(f"Loading data from {npz_filepath}...")
        data = np.load(npz_filepath)
        
        # Assuming the NPZ contains a single array named 'histograms'
        # Adjust 'histograms' if your array has a different key
        self.histograms = torch.from_numpy(data['histograms']).float() 
        
        # Expected shape: [2, M, E, a, e]
        # self.histograms[0] -> noisy1_data
        # self.histograms[1] -> noisy2_data
        
        if self.histograms.shape[0] != 2:
            raise ValueError(f"Expected the first dimension of data to be 2 (for noisy1 and noisy2), but got {self.histograms.shape[0]}")

        # --- NEW: Store min/max for normalization/denormalization ---
        # Calculate min and max across the entire dataset (both noisy_A and noisy_B)
        self.min_val = self.histograms.min().item()
        self.max_val = self.histograms.max().item()

        print(f"Original data range: [{self.min_val:.4f}, {self.max_val:.4f}]")

        # --- NEW: Normalize the entire histogram tensor ---
        if self.max_val > self.min_val:
            self.histograms = (self.histograms - self.min_val) / (self.max_val - self.min_val)
            print("Data normalized to [0, 1] range.")
        else:
            print("Warning: Max value is not greater than min value. Data not normalized (likely constant).")
            # If data is constant, set min/max for denormalization to prevent division by zero later
            self.min_val = 0.0 # Default if data is all zeros
            self.max_val = 1.0 if self.histograms.max().item() == 0 else self.histograms.max().item()


        self.noisy_full_volume_A = self.histograms[0] # This will be the input
        self.noisy_full_volume_B = self.histograms[1] # This will be the target

        # The full data dimensions are now derived from the loaded data
        # Skipping the first dimension (which is 2 for noisy A and B)
        self.full_data_dims = tuple(self.noisy_full_volume_A.shape) # (M, E, a, e)
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.signal_patch_ratio = signal_patch_ratio

        # Ensure patch_size matches the number of dimensions in full_data_dims (M, E, a, e)
        if len(self.patch_size) != len(self.full_data_dims):
            raise ValueError(f"patch_size ({len(self.patch_size)} dims) must have the same number of dimensions as full_data_dims ({len(self.full_data_dims)} dims). Expected {len(self.full_data_dims)} dimensions for patch_size.")

        # Ensure patch_size is not larger than full_data_dims in any dimension
        for i in range(len(self.full_data_dims)):
            if self.patch_size[i] > self.full_data_dims[i]:
                raise ValueError(f"Patch size dimension {i} ({self.patch_size[i]}) cannot be larger than full data dimension {i} ({self.full_data_dims[i]})")

        print(f"Loaded full data volume shape (M, E, a, e): {self.full_data_dims}")
        print(f"Patch size: {self.patch_size}")

        # For real data, 'clean_full_volume' doesn't exist, but we need a proxy for signal_coords
        # We'll consider any non-zero value in noisy_full_volume_A as "signal" for patch sampling
        print("Pre-computing coordinates of non-zero voxels for signal-centered patches from noisy_full_volume_A...")
        # IMPORTANT: Use the UNNORMALIZED data for identifying signal regions if your signal is defined by non-zero raw counts.
        # If your signal is just "any value above a very small normalized threshold", then use the normalized data.
        # For robustness, let's assume signal means original non-zero. If you normalize to [0,1], 0 will still be 0.
        # So using the normalized self.noisy_full_volume_A is fine here if original zeros map to zero.
        self.signal_coords = (self.noisy_full_volume_A != 0).nonzero(as_tuple=False).tolist() 
        print(f"Found {len(self.signal_coords)} non-zero (signal) voxels for patch centering.")
        if not self.signal_coords:
            print("Warning: No signal (non-zero voxels) found in the loaded data. All patches will be random.")
            # If no signal coords, set ratio to 0 to avoid errors
            self.signal_patch_ratio = 0.0


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        start_coords = [0] * len(self.full_data_dims)

        if random.random() < self.signal_patch_ratio and self.signal_coords:
            center_coord = random.choice(self.signal_coords)
            for d in range(len(self.full_data_dims)):
                potential_start = center_coord[d] - self.patch_size[d] // 2
                max_start = self.full_data_dims[d] - self.patch_size[d]
                start_coords[d] = max(0, min(potential_start, max_start))
        else:
            for d in range(len(self.full_data_dims)):
                start_coords[d] = torch.randint(0, self.full_data_dims[d] - self.patch_size[d] + 1, (1,)).item()

        # --- DEBUGGING PRINT STATEMENT (Optional, can be commented out) ---
        # print(f'Patch {idx}: Start Coordinates (D1, D2, D3, D4): {start_coords}')
        # end_coords = [start + p for start, p in zip(start_coords, self.patch_size)]
        # print(f'End Coordinates (D1, D2, D3, D4): {end_coords}')

        slices = tuple(slice(s, s + p) for s, p in zip(start_coords, self.patch_size))

        input_patch = self.noisy_full_volume_A[slices]
        target_patch = self.noisy_full_volume_B[slices]

        # Add the channel dimension (assuming 1 channel)
        # Shape becomes (1, M_patch, E_patch, a_patch, e_patch) for the patch
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

        # --- NEW: Add a final ReLU if your data is non-negative after denormalization ---
        # This prevents negative outputs from the network.
        self.final_activation = nn.ReLU(inplace=True) 

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.conv6(x) 
        # --- NEW: Apply final activation ---
        x = self.final_activation(x) 
        return x

def plot_multiple_d3d4_slices(input_vol: torch.Tensor, original_noisy_B_vol: torch.Tensor, output_vol: torch.Tensor,
                              full_data_dims: Tuple[int, ...], d1_indices: list, d2_indices: list, output_dir: str = '.',
                              min_val: float = 0.0, max_val: float = 1.0): # --- NEW: Added min_val, max_val
    """
    Plots D3-D4 slices of the 4D volumes at specified D1 and D2 indices.
    For real data, 'true_clean_vol' is replaced by 'original_noisy_B_vol' for comparison.

    Args:
        input_vol (torch.Tensor): The noisy input 4D volume (B=1, C=1, M, E, a, e) - Expected to be normalized.
        original_noisy_B_vol (torch.Tensor): The second noisy 4D volume used as target (B=1, C=1, M, E, a, e) - Expected to be normalized.
                                             Used here as a proxy for comparison if no clean data is available.
        output_vol (torch.Tensor): The denoised output 4D volume (B=1, C=1, M, E, a, e) - Expected to be normalized.
        full_data_dims (Tuple[int, ...]): Original full data dimensions (M, E, a, e).
        d1_indices (list): List of indices along the D1 (M) dimension to plot.
        d2_indices (list): List of indices along the D2 (E) dimension to plot.
        output_dir (str): Directory to save the plots.
        min_val (float): Original minimum value of the dataset for denormalization.
        max_val (float): Original maximum value of the dataset for denormalization.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nPlotting multiple D3-D4 slices in '{output_dir}'...")

    # --- NEW: Denormalization function ---
    def denormalize(tensor, min_val, max_val):
        if max_val > min_val:
            return tensor * (max_val - min_val) + min_val
        return tensor # If min_val == max_val, data was constant, no denormalization needed

    # Determine dimension names for clearer plotting
    dim_names = ['M', 'E', 'a', 'e']
    
    for d1_idx in d1_indices:
        if d1_idx >= full_data_dims[0] or d1_idx < 0:
            print(f"Warning: {dim_names[0]} index {d1_idx} is out of bounds for dimension size {full_data_dims[0]}. Skipping.")
            continue
        for d2_idx in d2_indices:
            if d2_idx >= full_data_dims[1] or d2_idx < 0:
                print(f"Warning: {dim_names[1]} index {d2_idx} is out of bounds for dimension size {full_data_dims[1]}. Skipping.")
                continue

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'4D Denoising - (a,e) Slice at {dim_names[0]}={d1_idx}, {dim_names[1]}={d2_idx}')

            # Extract 2D slices for plotting (fixed D1 and D2, varying D3 and D4)
            # Dims for plotting are D3 (a) and D4 (e)
            input_slice_normalized = input_vol[0, 0, d1_idx, d2_idx, :, :]
            target_noisy_slice_normalized = original_noisy_B_vol[0, 0, d1_idx, d2_idx, :, :]
            output_slice_normalized = output_vol[0, 0, d1_idx, d2_idx, :, :]

            # --- NEW: Denormalize the slices before plotting ---
            input_slice = denormalize(input_slice_normalized, min_val, max_val).cpu().numpy()
            target_noisy_slice = denormalize(target_noisy_slice_normalized, min_val, max_val).cpu().numpy()
            output_slice = denormalize(output_slice_normalized, min_val, max_val).cpu().numpy()


            # Determine global vmin/vmax for consistent color scaling across plots
            # Use the denormalized values for min/max
            all_slice_values = np.concatenate([input_slice.flatten(), target_noisy_slice.flatten(), output_slice.flatten()])
            global_vmin = all_slice_values.min()
            global_vmax = all_slice_values.max()
            # Add a small buffer for better visualization
            buffer = (global_vmax - global_vmin) * 0.05
            global_vmin -= buffer
            global_vmax += buffer

            im0 = axes[0].imshow(input_slice, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
            axes[0].set_title('Noisy Input (A)')
            fig.colorbar(im0, ax=axes[0])

            im1 = axes[1].imshow(target_noisy_slice, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
            axes[1].set_title('Noisy Target (B)') # Changed title for real data
            fig.colorbar(im1, ax=axes[1])

            im2 = axes[2].imshow(output_slice, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
            axes[2].set_title('Denoised Output')
            fig.colorbar(im2, ax=axes[2])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            filename = os.path.join(output_dir, f'4D_denoising_{dim_names[2]}{dim_names[3]}_slice_{dim_names[0]}_{d1_idx}_{dim_names[1]}_{d2_idx}.png')
            plt.savefig(filename)
            plt.close(fig) # Close the figure to free up memory
            print(f"Saved: {filename}")


def train_and_evaluate_4d_net_real_data(
    npz_filepath: str,
    num_epochs: int = 5,
    learning_rate: float = 0.0001,
    batch_size: int = 4,
    patch_size: Tuple[int, ...] = (16, 16, 16, 16),  # (M_patch, E_patch, a_patch, e_patch)
    num_patches_per_epoch: int = 500,  # How many patches to train on per epoch
    base_channels: int = 40,
    signal_patch_ratio: float = 0.8
):
    """
    Trains and evaluates the 4D convolutional neural network using real data from an NPZ file
    with the Noise2Noise paradigm and patching.
    """
    print(f"\n--- Training 4D Network on Real Data from '{npz_filepath}' on {device} ---")
    print(f"Patch Size for Training: {patch_size}")
    print(f"Base Channels: {base_channels}")
    print(f"Signal-centered patch ratio: {signal_patch_ratio * 100:.0f}%")

    # --- Create RealNoise2NoisePatchDataset and DataLoader ---
    dataset = RealNoise2NoisePatchDataset(
        npz_filepath=npz_filepath,
        patch_size=patch_size,
        num_samples=num_patches_per_epoch,
        signal_patch_ratio=signal_patch_ratio
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get the actual full data dimensions from the loaded dataset
    actual_full_data_dims = dataset.full_data_dims
    print(f"Actual Full Data Dimensions (M, E, a, e): {actual_full_data_dims}")

    # --- NEW: Get normalization values from the dataset ---
    dataset_min_val = dataset.min_val
    dataset_max_val = dataset.max_val
    print(f"Normalization values from dataset: min={dataset_min_val:.4f}, max={dataset_max_val:.4f}")


    print(f"Total patches (samples) per epoch: {num_patches_per_epoch}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches per epoch: {len(dataloader)}")

    kernel_size = 3

    model = Conv4dNet(in_channels=1, out_channels=1, kernel_size=kernel_size,
                      base_channels=base_channels).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} trainable parameters.")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
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

            pbar.set_postfix(
                {
                    "Batch Loss": f"{loss.item():.12f}",
                    "Avg Loss": f"{total_loss / (batch_idx + 1):.12f}"
                },
                refresh=True
            )

        avg_epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.12f}")

    print("Training finished.")

    # --- Evaluation and Plotting ---
    model.eval()
    with torch.no_grad():
        # Use the first noisy volume from the dataset for evaluation input
        evaluation_input = dataset.noisy_full_volume_A.unsqueeze(0).unsqueeze(0).to(device) # (B=1, C=1, M, E, a, e)
        # Use the second noisy volume as the "target for comparison" in plots
        original_noisy_B_vol = dataset.noisy_full_volume_B.unsqueeze(0).unsqueeze(0).to(device)

        print(f"\nPerforming inference on a full volume (shape {evaluation_input.shape}). This might take a moment...")
        start_time = time.time()
        evaluation_output = model(evaluation_input)
        end_time = time.time()

        print(f"Output shape: {evaluation_output.shape}")
        print(f"Inference time for full volume: {(end_time - start_time):.2f} seconds")

        # --- Plotting multiple (a,e) Slices ---
        # Choose a few representative indices for M and E
        # Use the actual dimensions obtained from the loaded data
        m_dim, e_dim, a_dim, e_dim_val = actual_full_data_dims # Unpack the dimensions

        selected_m_indices = sorted(list(set([
            0,
            m_dim // 4,
            m_dim // 2,
            (m_dim * 3) // 4,
            m_dim - 1
        ])))
        selected_m_indices = [idx for idx in selected_m_indices if 0 <= idx < m_dim]

        selected_e_indices = sorted(list(set([
            0,
            e_dim // 4,
            e_dim // 2,
            (e_dim * 3) // 4,
            e_dim - 1
        ])))
        selected_e_indices = [idx for idx in selected_e_indices if 0 <= idx < e_dim]


        plot_multiple_d3d4_slices(evaluation_input, original_noisy_B_vol, evaluation_output,
                                  actual_full_data_dims, selected_m_indices, selected_e_indices, 
                                  output_dir='denoising_real_data_slices',
                                  min_val=dataset_min_val, max_val=dataset_max_val) # --- NEW: Pass min/max values

if __name__ == "__main__":
    real_data_npz_filepath = 'DenoisingDataTransSheet.npz'
    if not os.path.exists(real_data_npz_filepath):
        print(f"NPZ file not found at: {real_data_npz_filepath}")
        print("Please ensure the file exists in the same directory or provide the full path.")
        exit()

    train_and_evaluate_4d_net_real_data(
        npz_filepath=real_data_npz_filepath,
        num_epochs=1, # Increased epochs for better learning
        learning_rate=0.001,
        batch_size=2,
        patch_size=(16, 16, 16, 16), # Keep patch size as suggested or adjust based on your real data
        num_patches_per_epoch=1000, # Increased patches per epoch
        base_channels=20, # Slightly adjusted base channels
        signal_patch_ratio=0.8
    )