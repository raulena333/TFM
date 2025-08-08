import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import random
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import time
from conv4d import Conv4d, BatchNorm4d, ConvTranspose4d

from scipy.ndimage import rotate as scipy_rotate # For 3D rotation

# Set device for PyTorch operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f'Python version: {sys.version}')
print(f'Pytorch version: {torch.__version__}')
torch.backends.cudnn.deterministic = True

# --- Your Data Loading Function ---
# This part is copied directly from your prompt.
PATCH_SIZE = (32, 32, 32, 32) # The order is M, E, a, e (L, D, H, W for Conv4d)
NUM_DIMS = len(PATCH_SIZE)

def load_and_prepare_data(npz_path="./DenoisingDataTransSheet.npz", required_patch_size = (32, 32, 32, 32)):
    '''
    Loads 5D data from an NPZ file,
    and generates a mask based on non-zero voxels in the noisy data.
    Adds Min-Max normalization.

    Args:
        npz_path (str): Path to the .npz file containing the 'histograms' data.
        required_patch_size (tuple): The minimum required patch size (M, E, a, e)
                                     for the U-Net architecture.

    Returns:
        tuple: (noisy_volume, target_volume, mask_volume)
               as PyTorch tensors.
    '''
    print(f"\nLoading data from: {npz_path}")
    try:
        loaded_data = np.load(npz_path)
        histograms_np = loaded_data['histograms']
        print(f"Loaded 'histograms' data with shape: {histograms_np.shape}")

        if histograms_np.ndim != 5 or histograms_np.shape[0] != 2:
            raise ValueError(
                "Expected 'histograms' data to be 5D with shape [2, M, E, a, e]. "
                f"Got shape: {histograms_np.shape}"
            )

        histograms_np = histograms_np.astype(np.float32)

        DATA_MIN_VAL = histograms_np.min()
        DATA_MAX_VAL = histograms_np.max()

        if DATA_MAX_VAL > DATA_MIN_VAL:
            histograms_np = (histograms_np - DATA_MIN_VAL) / (DATA_MAX_VAL - DATA_MIN_VAL)
            print(f"Data normalized to [0, 1] using min={DATA_MIN_VAL:.4f}, max={DATA_MAX_VAL:.4f}")
        else:
            histograms_np = np.zeros_like(histograms_np)
            print("Warning: Data is constant. Normalization resulted in all zeros.")
            
        noisy_data = histograms_np[0]
        target_data = histograms_np[1]
        
        vol_M, vol_E, vol_a, vol_e = noisy_data.shape
        patch_M, patch_E, patch_a, patch_e = required_patch_size
        
        if vol_M < patch_M or vol_E < patch_E or vol_a < patch_a or vol_e < patch_e:
            print(f"Warning: Loaded volume dimensions ({vol_M}, {vol_E}, {vol_a}, {vol_e}) "
                  f"are smaller than the required patch size ({patch_M}, {patch_E}, {patch_a}, {patch_e}).")
            print("This might lead to patches being entirely padding or unexpected behavior.")
            
        mask = (noisy_data > 0).astype(np.float32)

        print(f'Noisy volume 4D shape: {noisy_data.shape}')
        print(f'Target volume 4D shape: {target_data.shape}')
        print(f'Mask volume 4D shape: {mask.shape}')
        
        # Convert numpy arrays to Pytorch tensors and move to device
        # We don't add batch dimension here, Dataset will handle it.
        return torch.from_numpy(noisy_data), \
               torch.from_numpy(target_data), \
               torch.from_numpy(mask)
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {npz_path}")
    except Exception as e:
        raise Exception(f"Error loading data from {npz_path}: {e}")


# --- DenoisingDataset Class ---
class DenoisingDataset(Dataset):
    """
    A PyTorch Dataset for extracting 4D patches from noisy, target, and mask volumes.
    Implements a mixed sampling strategy: 80% patches centered on non-zero mask
    regions, 20% randomly sampled. Includes random 4D rotations for data augmentation,
    but only when patch dimensions are cubic (M=E=a=e).
    """
    def __init__(self, noisy_volume: torch.Tensor, target_volume: torch.Tensor, 
                 mask_volume: torch.Tensor, patch_size: tuple,
                 sampling_strategy_ratio: float = 0.8,
                 enable_rotations: bool = True, 
                 num_samples_per_epoch: int = 1000, # Added this parameter
                 device=torch.device('cpu')):
        """
        Args:
            noisy_volume (torch.Tensor): The 4D noisy data volume (M, E, a, e).
            target_volume (torch.Tensor): The 4D target data volume (M, E, a, e).
            mask_volume (torch.Tensor): The 4D mask volume (M, E, a, e), 1 for valid, 0 for invalid.
            patch_size (tuple): The size of the 4D patches to extract (M_p, E_p, a_p, e_p).
            sampling_strategy_ratio (float): Ratio of non-zero mask patches to random patches (0 to 1).
            enable_rotations (bool): Whether to apply random 4D rotations (only if patch_size is cubic).
            num_samples_per_epoch (int): The number of samples to yield per epoch.
            device (torch.device): Device to store tensors on.
        """
        super().__init__()
        self.noisy_volume = noisy_volume.to(device)
        self.target_volume = target_volume.to(device)
        self.mask_volume = mask_volume.to(device)
        self.patch_size = patch_size
        self.sampling_strategy_ratio = sampling_strategy_ratio
        self.enable_rotations = enable_rotations and all(d == patch_size[0] for d in patch_size) # Check for cubic
        self.num_samples_per_epoch = num_samples_per_epoch # Store the new parameter
        self.device = device

        if not all(d >= p for d, p in zip(self.noisy_volume.shape, self.patch_size)):
            raise ValueError(f"Patch size {self.patch_size} is larger than volume dimensions {self.noisy_volume.shape} in some axes.")

        self.volume_shape = self.noisy_volume.shape # (M, E, a, e)
        self.half_patch_size = tuple(p // 2 for p in self.patch_size)

        # Pre-compute valid non-zero mask coordinates for efficient sampling
        # Filter for coordinates where mask_volume is 1
        self.mask_coords = (self.mask_volume > 0).nonzero(as_tuple=False).cpu().numpy()
        print(f"Found {len(self.mask_coords)} non-zero voxels in mask for mask-based sampling.")

        if self.enable_rotations:
            print("4D rotations are enabled (patch size is cubic).")
        else:
            if not all(d == patch_size[0] for d in patch_size):
                print(f"4D rotations are disabled because patch size {patch_size} is not cubic.")
            else:
                print("4D rotations are disabled by setting enable_rotations=False.")

    def __len__(self):
        # Now returns the number of samples specified during initialization
        return self.num_samples_per_epoch

    def _get_random_patch_coords(self):
        # Generates random top-left corner (min_M, min_E, min_a, min_e) for a patch
        min_coords = [random.randint(0, dim - p_dim)
                      for dim, p_dim in zip(self.volume_shape, self.patch_size)]
        return min_coords

    def _get_mask_centered_patch_coords(self):
        if len(self.mask_coords) == 0:
            # Fallback to random if no mask points exist
            return self._get_random_patch_coords()

        # Randomly pick a non-zero voxel from the mask
        idx = random.randint(0, len(self.mask_coords) - 1)
        center_m, center_e, center_a, center_e = self.mask_coords[idx]

        # Calculate top-left corner to center the patch around this voxel
        min_m = center_m - self.half_patch_size[0]
        min_e = center_e - self.half_patch_size[1]
        min_a = center_a - self.half_patch_size[2]
        min_e = center_e - self.half_patch_size[3] # corrected variable name

        # Clamp coordinates to ensure patch stays within volume boundaries
        min_m = np.clip(min_m, 0, self.volume_shape[0] - self.patch_size[0])
        min_e = np.clip(min_e, 0, self.volume_shape[1] - self.patch_size[1])
        min_a = np.clip(min_a, 0, self.volume_shape[2] - self.patch_size[2])
        min_e = np.clip(min_e, 0, self.volume_shape[3] - self.patch_size[3])

        return [min_m, min_e, min_a, min_e]

    def _extract_patch(self, volume, min_coords):
        m, e, a, ee = min_coords
        pm, pe, pa, pee = self.patch_size
        return volume[m:m+pm, e:e+pe, a:a+pa, ee:ee+pee]

    def _random_4d_rotation(self, patch):
        # This function is now effectively unused, as augmentation is done in __getitem__
        # on the stacked patches to ensure consistency.
        if not self.enable_rotations:
            return patch

        patch_np = patch.cpu().numpy()
        permutation_axes = np.random.permutation(len(self.patch_size))
        patch_np = np.transpose(patch_np, axes=permutation_axes)

        for axis in range(len(self.patch_size)):
            if random.random() < 0.5:
                patch_np = np.flip(patch_np, axis=axis)

        return torch.from_numpy(patch_np.copy()).to(patch.dtype).to(self.device)


    def __getitem__(self, idx: int):
        # Mixed sampling strategy
        if random.random() < self.sampling_strategy_ratio:
            # 80% chance: sample patch centered on non-zero mask voxel
            patch_min_coords = self._get_mask_centered_patch_coords()
        else:
            # 20% chance: sample random patch
            patch_min_coords = self._get_random_patch_coords()

        # Extract patches
        noisy_patch = self._extract_patch(self.noisy_volume, patch_min_coords)
        target_patch = self._extract_patch(self.target_volume, patch_min_coords)
        mask_patch = self._extract_patch(self.mask_volume, patch_min_coords)

        # Apply augmentation if enabled and patch is cubic
        if self.enable_rotations:
            # Stack the patches along a new dimension (dim=0)
            combined_patch = torch.stack([noisy_patch, target_patch, mask_patch], dim=0) # Shape: (3, M, E, a, e)
            
            # Convert to numpy for augmentation operations (scipy/numpy are easier for this)
            combined_patch_np = combined_patch.cpu().numpy()

            # 1. Randomly permute spatial dimensions
            spatial_permutation = np.random.permutation(len(self.patch_size))
            new_axes_order = (0,) + tuple(sp_idx + 1 for sp_idx in spatial_permutation)
            combined_patch_np = np.transpose(combined_patch_np, axes=new_axes_order)

            # 2. Randomly flip along any spatial axis
            for axis_idx in range(1, 1 + len(self.patch_size)): # Iterate over spatial dimensions (indices 1 to 4)
                if random.random() < 0.5:
                    combined_patch_np = np.flip(combined_patch_np, axis=axis_idx)

            # Convert back to tensor and move to device
            augmented_combined_patch = torch.from_numpy(combined_patch_np.copy()).to(combined_patch.dtype).to(self.device)
            
            # Unstack the augmented patches
            noisy_patch, target_patch, mask_patch = augmented_combined_patch[0], augmented_combined_patch[1], augmented_combined_patch[2]

        # Add channel dimension (C=1 for all these volumes as they are single-channel)
        noisy_patch = noisy_patch.unsqueeze(0) # Shape: (1, M, E, a, e)
        target_patch = target_patch.unsqueeze(0) # Shape: (1, M, E, a, e)
        mask_patch = mask_patch.unsqueeze(0) # Shape: (1, M, E, a, e)
        
        return noisy_patch, target_patch, mask_patch, torch.tensor(patch_min_coords, dtype=torch.long)

# --- Simple Conv4d Network Definition --- 
class ComplexConv4dNet(nn.Module):
    """
    A more complex 4D Convolutional Neural Network for denoising,
    designed to output values between 0 and 1 (probabilities).
    It uses multiple Conv4d blocks with increasing and then decreasing channels,
    followed by BatchNorm4d and ReLU, and a final Sigmoid activation.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        """
        Initializes the ComplexConv4dNet.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel for all spatial dimensions.
            stride (int): Stride for the convolutional operation.
            padding (int): Padding for the convolutional operation.
        """
        super(ComplexConv4dNet, self).__init__()

        # Encoder-like path (increasing channels)
        self.conv_block1 = nn.Sequential(
            Conv4d(in_channels, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm4d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_block2 = nn.Sequential(
            Conv4d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm4d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder-like path (decreasing channels)
        self.conv_block3 = nn.Sequential(
            Conv4d(128, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm4d(64),
            nn.ReLU(inplace=True)
        )

        # Output layer with Sigmoid activation for probabilities
        self.output_layer = nn.Sequential(
            Conv4d(64, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Sigmoid() # Ensures output values are between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, D1, D2, D3, D4).

        Returns:
            torch.Tensor: Denoised output tensor of shape (N, C_out, D1', D2', D3', D4'),
                          with values between 0 and 1.
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.output_layer(x)
        return x
    
# --- Training Function ---
# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Trains and validates the given PyTorch model.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        num_epochs (int): The number of training epochs.
        device (torch.device): The device (CPU or GPU) to train on.

    Returns:
        tuple: (list of float, list of float) containing training and validation losses per epoch.
    """
    train_losses = []
    val_losses = []

    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_train_loss = 0.0
        # Wrap the train_loader with tqdm for a progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for batch_idx, (noisy_patch, target_patch, mask_patch, _) in enumerate(train_bar):
            noisy_patch = noisy_patch.to(device)
            target_patch = target_patch.to(device)
            # mask_patch = mask_patch.to(device) # Mask is not used in loss here, but could be for masked loss

            optimizer.zero_grad() # Zero the parameter gradients

            outputs = model(noisy_patch) # Forward pass

            # Calculate loss, potentially applying mask
            # For a simple denoising, we'll calculate MSE between output and target
            # If you want to apply the mask to the loss:
            # loss = criterion(outputs * mask_patch, target_patch * mask_patch)
            loss = criterion(outputs, target_patch)

            loss.backward() # Backward pass
            optimizer.step() # Optimize

            running_train_loss += loss.item()
            # Update tqdm postfix with current batch loss
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval() # Set model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad(): # No gradient calculation during validation
            # Wrap the val_loader with tqdm for a progress bar
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")
            for noisy_patch, target_patch, mask_patch, _ in val_bar:
                noisy_patch = noisy_patch.to(device)
                target_patch = target_patch.to(device)
                # mask_patch = mask_patch.to(device) # Mask not used in loss here

                outputs = model(noisy_patch)
                # val_loss = criterion(outputs * mask_patch, target_patch * mask_patch)
                val_loss = criterion(outputs, target_patch)
                running_val_loss += val_loss.item()
                # Update tqdm postfix with current batch validation loss
                val_bar.set_postfix(val_loss=f"{val_loss.item():.4f}")
        
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    print("\nTraining complete!")
    return train_losses, val_losses

# --- Full Volume Denoising Function ---
def denoise_full_volume(model: nn.Module, noisy_volume: torch.Tensor, patch_size: tuple, device: torch.device):
    """
    Denoises an entire 4D volume by processing it in patches.

    Args:
        model (nn.Module): The trained denoising model.
        noisy_volume (torch.Tensor): The 4D noisy input volume (M, E, a, e).
        patch_size (tuple): The size of the 4D patches (M_p, E_p, a_p, e_p) the model expects.
        device (torch.device): The device (CPU or GPU) to perform inference on.

    Returns:
        torch.Tensor: The denoised 4D volume of the same spatial dimensions as noisy_volume.
    """
    model.eval() # Set model to evaluation mode
    
    volume_shape = noisy_volume.shape # (M, E, a, e)
    denoised_volume = torch.zeros_like(noisy_volume, device=device)
    
    # Calculate padding needed to make volume dimensions divisible by patch size
    padding_needed = [
        (ps - (vs % ps)) % ps for vs, ps in zip(volume_shape, patch_size)
    ]
    
    # Pad the noisy volume
    # F.pad expects (padding_left, padding_right, padding_top, padding_bottom, ...)
    # For 4D, it's (pad_e_left, pad_e_right, pad_a_left, pad_a_right, pad_E_left, pad_E_right, pad_M_left, pad_M_right)
    # We need to reverse the order of padding_needed for F.pad
    pad_dims = []
    for p in reversed(padding_needed):
        pad_dims.extend([0, p]) # (start_pad, end_pad) for each dimension

    # Add channel and batch dimensions for padding function
    padded_noisy_volume = F.pad(noisy_volume.unsqueeze(0).unsqueeze(0), pad_dims, mode='constant', value=0)
    padded_noisy_volume = padded_noisy_volume.squeeze(0).squeeze(0) # Remove added dims

    print(f"Padded noisy volume shape: {padded_noisy_volume.shape}")

    # Iterate through the padded volume in steps of patch_size
    # This assumes non-overlapping patches for simplicity in reconstruction
    # For better results with overlapping patches, a more complex overlap-add
    # or weighted averaging scheme would be required.
    
    # Define ranges for iteration
    ranges = [range(0, dim, ps) for dim, ps in zip(padded_noisy_volume.shape, patch_size)]

    total_patches = 1
    for r in ranges:
        total_patches *= len(r)

    print(f"Denoising full volume with {total_patches} patches...")
    
    with torch.no_grad():
        for m_start in tqdm(ranges[0], desc="Denoising M"):
            for e_start in ranges[1]:
                for a_start in ranges[2]:
                    for ee_start in ranges[3]:
                        # Extract patch
                        current_patch = padded_noisy_volume[
                            m_start : m_start + patch_size[0],
                            e_start : e_start + patch_size[1],
                            a_start : a_start + patch_size[2],
                            ee_start : ee_start + patch_size[3]
                        ].to(device)

                        # Add batch and channel dimensions (1, 1, M, E, a, e)
                        current_patch = current_patch.unsqueeze(0).unsqueeze(0)

                        # Denoise the patch
                        denoised_patch = model(current_patch)

                        # Remove batch and channel dimensions
                        denoised_patch = denoised_patch.squeeze(0).squeeze(0)

                        # Place denoised patch into the reconstructed volume
                        # This simple assignment works for non-overlapping patches
                        denoised_volume[
                            m_start : m_start + patch_size[0],
                            e_start : e_start + patch_size[1],
                            a_start : a_start + patch_size[2],
                            ee_start : ee_start + patch_size[3]
                        ] = denoised_patch.cpu() # Move back to CPU for final volume

    # Crop the denoised volume back to original size if padding was applied
    denoised_volume_cropped = denoised_volume[
        :volume_shape[0],
        :volume_shape[1],
        :volume_shape[2],
        :volume_shape[3]
    ]
    print("Full volume denoising complete.")
    return denoised_volume_cropped
    
# --- Main Execution Block ---
if __name__ == "__main__":
    # Create a dummy .npz file if it doesn't exist
    dummy_npz_path = "./DenoisingDataTransSheet.npz"
    if not os.path.exists(dummy_npz_path):
        raise FileNotFoundError(f"Dummy .npz file not found at {dummy_npz_path}")

    # --- Continue with DenoisingDataset and DataLoader tests ---
    try:
        noisy_volume, target_volume, mask_volume = load_and_prepare_data(
            npz_path=dummy_npz_path,
            required_patch_size=PATCH_SIZE
        )

        num_total_patches = 1000
        denoising_dataset = DenoisingDataset(
            noisy_volume=noisy_volume,
            target_volume=target_volume,
            mask_volume=mask_volume,
            patch_size=PATCH_SIZE,
            sampling_strategy_ratio=0.8,
            enable_rotations=True,
            num_samples_per_epoch=num_total_patches,
            device=device
        )

        print(f"\nDataset length (cubic patch): {len(denoising_dataset)}")
        
        train_size = int(0.8 * len(denoising_dataset))
        val_size = len(denoising_dataset) - train_size
        train_dataset, val_dataset = random_split(denoising_dataset, [train_size, val_size])

        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        # --- Test Data Loader ---
        print("\nTesting DataLoader ...")
        batch_size = 2
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        
        model = ComplexConv4dNet(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        print(model)
        print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        num_epochs = 1
        train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
        
        # --- Denoise and Plot Full Volume Slices ---
        print("\nDenoising the entire 4D histogram...")
        denoised_full_volume_normalized = denoise_full_volume(model, noisy_volume, PATCH_SIZE, device)
        print(f"Denoised full volume (normalized) shape: {denoised_full_volume_normalized.shape}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure '{dummy_npz_path}' is accessible or replace with your actual data path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")