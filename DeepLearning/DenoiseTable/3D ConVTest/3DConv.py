import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import random
from tqdm import tqdm
import torch.optim as optim
import os
import matplotlib.pyplot as plt # Added for plotting

# Global variables
PATCH_SIZE = (32, 32, 128) # Patch size for training (example: non-cubic)
ORIGINAL_A_DIM = None     # Will store original 'a' dimension from 5D data
ORIGINAL_E_DIM = None     # Will store original 'e' dimension from 5D data
DATA_MIN_VAL = None       # Will store global min for denormalization
DATA_MAX_VAL = None       # Will store global max for denormalization

def load_and_prepare_data(npz_path="./DenoisingDataTransSheet50.npz", required_patch_size=(32, 32, 32)):
    """
    Loads 5D data from an NPZ file, flattens it to 3D volumes,
    generates a mask, and *creates a 3-channel coordinate volume* for positional encoding.
    Adds Min-Max normalization.

    Args:
        npz_path (str): Path to the .npz file containing the 'histograms' data.
        required_patch_size (tuple): The minimum required patch size (D, H, W)
                                     for the U-Net architecture.

    Returns:
        tuple: (noisy_volume_3d, target_volume_3d, mask_volume_3d, coordinate_volume_3d, original_a, original_e)
               as PyTorch tensors and integers.
    """
    global ORIGINAL_A_DIM, ORIGINAL_E_DIM, DATA_MIN_VAL, DATA_MAX_VAL # Declare intent to modify global variables

    print(f"Loading data from: {npz_path}")
    try:
        loaded_data = np.load(npz_path)
        histograms_np = loaded_data['histograms'].astype(np.float32)
        print(f"Loaded 'histograms' data with shape: {histograms_np.shape}")

        if histograms_np.ndim != 5 or histograms_np.shape[0] != 2:
            raise ValueError(
                "Expected 'histograms' data to be 5D with shape [2, M, E, a, e]. "
                f"Got shape: {histograms_np.shape}"
            )

        # --- Normalization ---
        DATA_MIN_VAL = histograms_np.min()
        DATA_MAX_VAL = histograms_np.max()

        if DATA_MAX_VAL > DATA_MIN_VAL:
            histograms_np = (histograms_np - DATA_MIN_VAL) / (DATA_MAX_VAL - DATA_MIN_VAL)
            print(f"Data normalized to [0, 1] using min={DATA_MIN_VAL:.4f}, max={DATA_MAX_VAL:.4f}")
        else:
            histograms_np = np.zeros_like(histograms_np)
            print("Warning: Data is constant. Normalization resulted in all zeros.")

        noisydata1_4d = histograms_np[0] # Shape: [M, E, a, e]
        noisydata2_4d = histograms_np[1] # Shape: [M, E, a, e] (Target)

        M_dim, E_dim, a_dim, e_dim = noisydata1_4d.shape
        ORIGINAL_A_DIM = a_dim
        ORIGINAL_E_DIM = e_dim

        noisydata1_3d = noisydata1_4d.reshape(M_dim, E_dim, a_dim * e_dim)
        noisydata2_3d = noisydata2_4d.reshape(M_dim, E_dim, a_dim * e_dim)

        vol_D, vol_H, vol_W = noisydata1_3d.shape
        patch_D, patch_H, patch_W = required_patch_size
        if vol_D < patch_D or vol_H < patch_H or vol_W < patch_W:
            print(f"Warning: Loaded volume dimensions ({vol_D}, {vol_H}, {vol_W}) "
                  f"are smaller than the required patch size ({patch_D}, {patch_H}, {patch_W}).")
            print("This might lead to patches being entirely padding or unexpected behavior.")

        mask = (noisydata1_3d != 0)

        # --- Generate Positional Encoding (Coordinate Volume) ---
        # Create normalized coordinate grids for D, H, W dimensions
        coords_d = torch.linspace(0, 1, vol_D).reshape(vol_D, 1, 1).expand(vol_D, vol_H, vol_W)
        coords_h = torch.linspace(0, 1, vol_H).reshape(1, vol_H, 1).expand(vol_D, vol_H, vol_W)
        coords_w = torch.linspace(0, 1, vol_W).reshape(1, 1, vol_W).expand(vol_D, vol_H, vol_W)

        # Stack them to create a (3, D, H, W) coordinate volume
        coordinate_volume_3d = torch.stack((coords_d, coords_h, coords_w), dim=0).float()
        print(f"Coordinate Volume Shape: {coordinate_volume_3d.shape}")


        print(f"Noisy Volume (3D) Shape: {noisydata1_3d.shape}")
        print(f"Target Volume (3D) Shape: {noisydata2_3d.shape}")
        print(f"Mask Volume (3D) Shape: {mask.shape} (derived from non-zero voxels)")

        # Convert NumPy arrays to PyTorch tensors
        return torch.from_numpy(noisydata1_3d), torch.from_numpy(noisydata2_3d), \
               torch.from_numpy(mask), coordinate_volume_3d, a_dim, e_dim

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {npz_path}")
    except Exception as e:
        raise Exception(f"Error loading data from {npz_path}: {e}")


class DenoisingDataset(Dataset):
    """
    A PyTorch Dataset for extracting 3D patches from noisy, target, mask, and coordinate volumes.
    Implements a mixed sampling strategy: 80% patches centered on non-zero mask
    regions, 20% randomly sampled. Includes random 3D rotations for data augmentation,
    but only when patch dimensions are cubic.
    """
    def __init__(self, noisy_volume, target_volume, mask_volume, coordinate_volume, patch_size, num_patches=1000):
        """
        Args:
            noisy_volume (torch.Tensor): The 3D noisy data volume (D, H, W).
            target_volume (torch.Tensor): The 3D target data volume (D, H, W).
            mask_volume (torch.Tensor): The 3D boolean mask volume (D, H, W), True for non-zero.
            coordinate_volume (torch.Tensor): The 3D coordinate volume (3, D, H, W).
            patch_size (tuple): The desired patch size (patch_D, patch_H, patch_W).
            num_patches (int): The total number of patches to generate for the dataset.
        """
        self.noisy_volume = noisy_volume
        self.target_volume = target_volume
        self.mask_volume = mask_volume
        self.coordinate_volume = coordinate_volume # New: Store coordinate volume
        self.patch_D, self.patch_H, self.patch_W = patch_size
        self.volume_D, self.volume_H, self.volume_W = noisy_volume.shape
        self.num_patches = num_patches

        self.valid_non_zero_centers = self._get_valid_non_zero_centers()
        print(f"Found {len(self.valid_non_zero_centers)} valid non-zero centers for patch extraction.")

        self.patch_coords = self._generate_patch_coordinates()
        print(f"Generated {len(self.patch_coords)} patch coordinates.")

    def _get_valid_non_zero_centers(self):
        """
        Finds all (d, h, w) coordinates in the mask_volume that are True.
        We now rely entirely on the __getitem__ padding to handle patch boundaries.
        """
        non_zero_indices = torch.nonzero(self.mask_volume, as_tuple=False)
        # Simply convert all non-zero indices to a list of lists/tuples
        valid_centers = [idx.tolist() for idx in tqdm(non_zero_indices, desc="Collecting all non-zero centers")]
        return valid_centers

    def _generate_patch_coordinates(self):
        # (Same as before)
        coords = []
        num_non_zero = int(self.num_patches * 0.8)
        num_random = self.num_patches - num_non_zero

        if not self.valid_non_zero_centers:
            print("Warning: No valid non-zero centers found. All patches will be random.")
            num_random = self.num_patches
            num_non_zero = 0

        if num_non_zero > 0:
            if num_non_zero > len(self.valid_non_zero_centers):
                print(f"Warning: Requested {num_non_zero} non-zero patches, but only {len(self.valid_non_zero_centers)} valid centers available. Sampling with replacement.")
                coords.extend(random.choices(self.valid_non_zero_centers, k=num_non_zero))
            else:
                coords.extend(random.sample(self.valid_non_zero_centers, k=num_non_zero))

        for _ in tqdm(range(num_random), desc="Generating random patch centers"):
            coords.append(self._get_random_patch_center())

        random.shuffle(coords)
        return coords

    def _get_random_patch_center(self):
        # (Same as before)
        d = random.randint(0, self.volume_D - 1)
        h = random.randint(0, self.volume_H - 1)
        w = random.randint(0, self.volume_W - 1)
        return (d, h, w)

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
        """
        Extracts a patch from the noisy, target, mask, and coordinate volumes given a center coordinate.
        Handles padding and applies random 3D rotations (only if patch is cubic).
        """
        center_d, center_h, center_w = self.patch_coords[idx]

        half_patch_D = self.patch_D // 2
        half_patch_H = self.patch_H // 2
        half_patch_W = self.patch_W // 2

        start_d = center_d - half_patch_D
        end_d = start_d + self.patch_D
        start_h = center_h - half_patch_H
        end_h = start_h + self.patch_H
        start_w = center_w - half_patch_W
        end_w = start_w + self.patch_W

        pad_left_d = max(0, -start_d)
        pad_right_d = max(0, end_d - self.volume_D)
        pad_left_h = max(0, -start_h)
        pad_right_h = max(0, end_h - self.volume_H)
        pad_left_w = max(0, -start_w)
        pad_right_w = max(0, end_w - self.volume_W)

        slice_start_d = max(0, start_d)
        slice_end_d = min(self.volume_D, end_d)
        slice_start_h = max(0, start_h)
        slice_end_h = min(self.volume_H, end_h)
        slice_start_w = max(0, start_w)
        slice_end_w = min(self.volume_W, end_w)

        noisy_patch_slice = self.noisy_volume[
            slice_start_d:slice_end_d,
            slice_start_h:slice_end_h,
            slice_start_w:slice_end_w
        ]
        target_patch_slice = self.target_volume[
            slice_start_d:slice_end_d,
            slice_start_h:slice_end_h,
            slice_start_w:slice_end_w
        ]
        mask_patch_slice = self.mask_volume[
            slice_start_d:slice_end_d,
            slice_start_h:slice_end_h,
            slice_start_w:slice_end_w
        ]
        # New: Extract coordinate patch slice - it's channel-first (3, D, H, W)
        coordinate_patch_slice = self.coordinate_volume[
            :, # All 3 channels
            slice_start_d:slice_end_d,
            slice_start_h:slice_end_h,
            slice_start_w:slice_end_w
        ]

        padding = (pad_left_w, pad_right_w,
                   pad_left_h, pad_right_h,
                   pad_left_d, pad_right_d)

        # Reverted padding mode to 'constant', value=0 based on user's environment
        noisy_patch = F.pad(noisy_patch_slice, padding, mode='constant', value=0)
        target_patch = F.pad(target_patch_slice, padding, mode='constant', value=0)
        mask_patch = F.pad(mask_patch_slice.float(), padding, mode='constant', value=0).bool()
        coordinate_patch = F.pad(coordinate_patch_slice, padding, mode='constant', value=0)


        # --- Data Augmentation: Random 3D Rotations ---
        # Only apply rotations if the patch is cubic to avoid dimension mismatches
        if self.patch_D == self.patch_H == self.patch_W:
            axis_to_rotate = random.choice([0, 1, 2])
            k_rotations = random.randint(0, 3)

            if k_rotations > 0:
                if axis_to_rotate == 0: # Rotate around D-axis (rotate H, W plane)
                    dims = (1, 2) # For (D, H, W) data
                    coord_dims = (2, 3) # For (C, D, H, W) coord, C is dim 0
                elif axis_to_rotate == 1: # Rotate around H-axis (rotate D, W plane)
                    dims = (0, 2)
                    coord_dims = (1, 3)
                else: # axis_to_rotate == 2, Rotate around W-axis (rotate D, H plane)
                    dims = (0, 1)
                    coord_dims = (1, 2)
                
                noisy_patch = torch.rot90(noisy_patch, k=k_rotations, dims=dims)
                target_patch = torch.rot90(target_patch, k=k_rotations, dims=dims)
                mask_patch = torch.rot90(mask_patch, k=k_rotations, dims=dims)
                coordinate_patch = torch.rot90(coordinate_patch, k=k_rotations, dims=coord_dims)
        # else:
            # print(f"Skipping 3D rotations for non-cubic patch size: {self.patch_D, self.patch_H, self.patch_W}") # Uncomment for debugging


        # Assertions to ensure the patches have the correct final size
        assert noisy_patch.shape == (self.patch_D, self.patch_H, self.patch_W), \
            f"Noisy patch shape mismatch after rotation: Expected {self.patch_D, self.patch_H, self.patch_W}, Got {noisy_patch.shape}"
        assert target_patch.shape == (self.patch_D, self.patch_H, self.patch_W), \
            f"Target patch shape mismatch after rotation: Expected {self.patch_D, self.patch_H, self.patch_W}, Got {target_patch.shape}"
        assert mask_patch.shape == (self.patch_D, self.patch_H, self.patch_W), \
            f"Mask patch shape mismatch after rotation: Expected {self.patch_D, self.patch_H, self.patch_W}, Got {mask_patch.shape}"
        assert coordinate_patch.shape == (3, self.patch_D, self.patch_H, self.patch_W), \
            f"Coord patch shape mismatch after rotation: Expected (3, {self.patch_D, self.patch_H, self.patch_W}), Got {coordinate_patch.shape}"

        # Return the data patch, target, mask, and the coordinate patch separately.
        # The main training loop will concatenate noisy_patch and coordinate_patch.
        return noisy_patch, target_patch, mask_patch, coordinate_patch

# Squeeze-and-Excite block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

# Basic Conv3D block with optional SE
class ConvBlock3D(nn.Module):
    def __init__(self, in_c, out_c, se=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_c) if se else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        return self.se(x)

# Full 3D U-Net
class Noise2Noise3DUNet(nn.Module):
    def __init__(self, in_channels=4, base_channels=32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, base_channels, se=True)
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2, se=True)
        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4, se=True)
        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock3D(base_channels * 4, base_channels * 8, se=True)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_channels * 2, base_channels)

        # Output layer: predict noise
        self.out = nn.Conv3d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)         # [B, C, D, H, W]
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        # Bottleneck
        x4 = self.bottleneck(self.pool(x3))

        # Decoder
        x = self.up3(x4)
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        # Predict noise
        noise = self.out(x)

        return noise

def total_variation_loss(x):
    """Computes total variation loss in 3D"""
    tv_z = torch.mean(torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]))
    tv_y = torch.mean(torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]))
    tv_x = torch.mean(torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]))
    return tv_z + tv_y + tv_x

def train_noise2noise(
    model,
    train_loader,
    val_loader,
    num_epochs,
    lr=1e-5,
    device='cuda',
    loss_type="mse+l1+tv"
):
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-7
    )

    # Loss functions
    mse_loss = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)

    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, (noisy_patch, target_patch, mask_patch, coord_patch) in train_bar:
            noisy_patch = noisy_patch.unsqueeze(1).to(device)
            target_patch = target_patch.unsqueeze(1).to(device)
            coord_patch = coord_patch.to(device)
            input_patch = torch.cat([noisy_patch, coord_patch], dim=1)

            optimizer.zero_grad()
            output = model(input_patch)

            # Base loss
            loss = mse_loss(output, target_patch)

            # Add optional L1
            if "l1" in loss_type:
                loss += 0.1 * l1_loss(output, target_patch)

            # Add optional Total Variation (TV)
            if "tv" in loss_type:
                loss += 0.01 * total_variation_loss(output)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_batch_loss = running_loss / (batch_idx + 1)
            train_bar.set_postfix(loss=avg_batch_loss)

        avg_train_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Train Loss: {avg_train_loss:.6f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch+1} [Val]")
        with torch.no_grad():
            for batch_idx, (noisy_patch, target_patch, mask_patch, coord_patch) in val_bar:
                noisy_patch = noisy_patch.unsqueeze(1).to(device)
                target_patch = target_patch.unsqueeze(1).to(device)
                coord_patch = coord_patch.to(device)
                input_patch = torch.cat([noisy_patch, coord_patch], dim=1)

                output = model(input_patch)
                loss = mse_loss(output, target_patch)

                if "l1" in loss_type:
                    loss += 0.1 * l1_loss(output, target_patch)
                if "tv" in loss_type:
                    loss += 0.01 * total_variation_loss(output)

                val_loss += loss.item()
                avg_val_batch_loss = val_loss / (batch_idx + 1)
                val_bar.set_postfix(loss=avg_val_batch_loss)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.6f}")

        # Step scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.2e}")

def denoise_volume(
    model: nn.Module,
    noisy_volume: torch.Tensor,
    coordinate_volume: torch.Tensor,
    patch_size: tuple,
    overlap_ratio: float = 0.5,
    batch_size: int = 1,
    device: str = 'cuda'
) -> torch.Tensor: # Updated return type hint to only Tensor
    """
    Denoises a full 3D volume using a trained Noise2Noise3DUNet model.
    Processes the volume in overlapping patches and averages the results.

    Args:
        model (nn.Module): The trained Noise2Noise3DUNet model.
        noisy_volume (torch.Tensor): The 3D noisy data volume (D, H, W). Assumed to be normalized [0, 1].
        coordinate_volume (torch.Tensor): The 3D coordinate volume (3, D, H, W).
        patch_size (tuple): The (D, H, W) size of patches the model was trained on.
        overlap_ratio (float): The ratio of overlap between adjacent patches (e.g., 0.5 for 50% overlap).
                               Must be between 0.0 and 1.0.
        batch_size (int): Batch size for inference. Larger batch sizes might be faster but use more memory.
        device (str): The device to perform inference on ('cuda' or 'cpu').

    Returns:
        torch.Tensor: The denoised 4D volume (M, E, a, e), denormalized and reshaped to the original 5D data format.
    """
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be between 0.0 (inclusive) and 1.0 (exclusive).")

    model.eval() # Set model to evaluation mode
    model.to(device)

    vol_D, vol_H, vol_W = noisy_volume.shape
    patch_D, patch_H, patch_W = patch_size

    # Calculate strides for sliding window
    # Ensure stride is at least 1
    step_d = max(1, int(patch_D * (1 - overlap_ratio)))
    step_h = max(1, int(patch_H * (1 - overlap_ratio)))
    step_w = max(1, int(patch_W * (1 - overlap_ratio)))

    # Initialize output volume and overlap counter
    denoised_full_volume = torch.zeros_like(noisy_volume, dtype=torch.float32)
    overlap_counter = torch.zeros_like(noisy_volume, dtype=torch.float32)

    # Lists to store patches and their corresponding coordinates for batch processing
    input_patches_list = []
    output_coords_list = [] # Store coordinates of where to place the output patch in the full volume

    # Iterate through the volume to extract patches
    # tqdm for outer loops to show progress for large volumes
    for d_start in tqdm(range(0, vol_D, step_d), desc="Extracting Denoising Patches (Depth)"):
        for h_start in range(0, vol_H, step_h):
            for w_start in range(0, vol_W, step_w):
                # Define slice for the current patch
                d_end = min(d_start + patch_D, vol_D)
                h_end = min(h_start + patch_H, vol_H)
                w_end = min(w_start + patch_W, vol_W)

                noisy_sub_volume = noisy_volume[d_start:d_end, h_start:h_end, w_start:w_end]
                coord_sub_volume = coordinate_volume[:, d_start:d_end, h_start:h_end, w_start:w_end]

                # Calculate padding needed for this specific patch to reach `patch_size`
                pad_d = patch_D - noisy_sub_volume.shape[0]
                pad_h = patch_H - noisy_sub_volume.shape[1]
                pad_w = patch_W - noisy_sub_volume.shape[2]

                # Pad the noisy and coordinate patches
                # F.pad expects padding in (W_left, W_right, H_left, H_right, D_left, D_right) order
                padding = (0, pad_w, 0, pad_h, 0, pad_d)
                
                padded_noisy_patch = F.pad(noisy_sub_volume, padding, mode='constant', value=0)
                padded_coord_patch = F.pad(coord_sub_volume, padding, mode='constant', value=0)

                # Add channel dimension for noisy_patch (1, D, H, W)
                padded_noisy_patch = padded_noisy_patch.unsqueeze(0) # Becomes (1, D, H, W)

                # Concatenate noisy patch (1 channel) and coordinate patch (3 channels)
                # Resulting input_patch will be (4, D, H, W)
                input_patch = torch.cat([padded_noisy_patch, padded_coord_patch], dim=0)
                
                input_patches_list.append(input_patch)
                output_coords_list.append(((d_start, d_end), (h_start, h_end), (w_start, w_end)))

                # Process in batches
                if len(input_patches_list) == batch_size:
                    batch_input = torch.stack(input_patches_list).to(device) # Shape: (B, C, D, H, W)
                    
                    with torch.no_grad():
                        denoised_batch_output = model(batch_input) # Output shape: (B, 1, D, H, W)

                    # Place denoised patches back into the full volume
                    for i in range(batch_input.shape[0]):
                        out_patch = denoised_batch_output[i, 0] # Remove batch and channel dims: (D, H, W)
                        
                        # Crop the output patch to the original (unpadded) size
                        current_d_slice, current_h_slice, current_w_slice = output_coords_list[i]
                        
                        # Calculate actual dimensions of the original sub-volume
                        actual_d_dim = current_d_slice[1] - current_d_slice[0]
                        actual_h_dim = current_h_slice[1] - current_h_slice[0]
                        actual_w_dim = current_w_slice[1] - current_w_slice[0]

                        denoised_cropped_patch = out_patch[:actual_d_dim, :actual_h_dim, :actual_w_dim]
                        
                        # Add to the full volume
                        denoised_full_volume[
                            current_d_slice[0]:current_d_slice[1],
                            current_h_slice[0]:current_h_slice[1],
                            current_w_slice[0]:current_w_slice[1]
                        ] += denoised_cropped_patch.cpu() # Move back to CPU for accumulation

                        # Increment overlap counter
                        overlap_counter[
                            current_d_slice[0]:current_d_slice[1],
                            current_h_slice[0]:current_h_slice[1],
                            current_w_slice[0]:current_w_slice[1]
                        ] += 1

                    input_patches_list = [] # Clear the list for the next batch
                    output_coords_list = []

    # Process any remaining patches
    if len(input_patches_list) > 0:
        batch_input = torch.stack(input_patches_list).to(device)
        with torch.no_grad():
            denoised_batch_output = model(batch_input)

        for i in range(batch_input.shape[0]):
            out_patch = denoised_batch_output[i, 0]
            current_d_slice, current_h_slice, current_w_slice = output_coords_list[i]
            
            actual_d_dim = current_d_slice[1] - current_d_slice[0]
            actual_h_dim = current_h_slice[1] - current_h_slice[0]
            actual_w_dim = current_w_slice[1] - current_w_slice[0]

            denoised_cropped_patch = out_patch[:actual_d_dim, :actual_h_dim, :actual_w_dim]

            denoised_full_volume[
                current_d_slice[0]:current_d_slice[1],
                current_h_slice[0]:current_h_slice[1],
                current_w_slice[0]:current_w_slice[1]
            ] += denoised_cropped_patch.cpu()

            overlap_counter[
                current_d_slice[0]:current_d_slice[1],
                current_h_slice[0]:current_h_slice[1],
                current_w_slice[0]:current_w_slice[1]
            ] += 1

    # Final averaging
    # Handle cases where overlap_counter might be zero (shouldn't happen with proper strides/padding)
    overlap_counter[overlap_counter == 0] = 1 # Prevent division by zero
    denoised_full_volume /= overlap_counter

    # Denormalize the output
    if DATA_MIN_VAL is not None and DATA_MAX_VAL is not None:
        denoised_full_volume = denoised_full_volume * (DATA_MAX_VAL - DATA_MIN_VAL) + DATA_MIN_VAL
        print(f"Denoised volume denormalized using min={DATA_MIN_VAL:.4f}, max={DATA_MAX_VAL:.4f}")
    else:
        print("Warning: DATA_MIN_VAL or DATA_MAX_VAL not set. Output not denormalized.")

    # Reshape back to 4D: [M, E, a, e] from [D, H, W] where W = a*e
    # D corresponds to M_dim, H to E_dim, and W splits into a_dim, e_dim
    if ORIGINAL_A_DIM is not None and ORIGINAL_E_DIM is not None:
        denoised_4d_volume = denoised_full_volume.reshape(
            vol_D, vol_H, ORIGINAL_A_DIM, ORIGINAL_E_DIM
        )
        print(f"Denoised volume reshaped to 4D: {denoised_4d_volume.shape}")
    else:
        denoised_4d_volume = None
        print("Warning: ORIGINAL_A_DIM or ORIGINAL_E_DIM not set. Cannot reshape to 4D.")
        # If 4D reshape fails, return the 3D volume as a fallback, or raise an error
        # For now, returning None if reshape fails, but you might want to adjust this.

    # Only return the 4D volume as requested
    return denoised_4d_volume

def plot_2d_histograms(
    noisy_volume_3d: torch.Tensor,
    target_volume_3d: torch.Tensor,
    denoised_volume_4d: torch.Tensor,
    m_idx: int,
    e_idx: int,
    original_a_dim: int,
    original_e_dim: int,
    data_min_val: float,
    data_max_val: float,
    angleRange = (0,70),
    energyRange = (-0.6, 0)
):
    """
    Plots 2D histograms comparing initial noisy, target (noisy2), and denoised data
    for a specific M and E index.

    Args:
        noisy_volume_3d (torch.Tensor): The 3D noisy data volume (M, E, a*e), normalized.
        target_volume_3d (torch.Tensor): The 3D target data volume (M, E, a*e), normalized.
        denoised_volume_4d (torch.Tensor): The 4D denoised volume (M, E, a, e), denormalized.
        m_idx (int): The index for the M dimension (depth/first dimension).
        e_idx (int): The index for the E dimension (height/second dimension).
        original_a_dim (int): The original 'a' dimension from the 5D data.
        original_e_dim (int): The original 'e' dimension from the 5D data.
        data_min_val (float): The global minimum value used for original data normalization.
        data_max_val (float): The global maximum value used for original data normalization.
    """
    # Ensure indices are within bounds
    if not (0 <= m_idx < noisy_volume_3d.shape[0] and
            0 <= e_idx < noisy_volume_3d.shape[1]):
        print(f"Error: M index ({m_idx}) or E index ({e_idx}) out of bounds for volumes of shape {noisy_volume_3d.shape[:2]}.")
        return

    # Denormalize noisy_volume_3d and target_volume_3d for fair comparison
    if data_max_val > data_min_val:
        noisy_volume_3d_denormalized = noisy_volume_3d * (data_max_val - data_min_val) + data_min_val
        target_volume_3d_denormalized = target_volume_3d * (data_max_val - data_min_val) + data_min_val
    else:
        # If data was constant, denormalization results in zeros, use as is
        noisy_volume_3d_denormalized = noisy_volume_3d
        target_volume_3d_denormalized = target_volume_3d
        print("Warning: Data was constant, denormalization for plotting inputs might not be meaningful.")

    # Reshape 3D volumes to 4D for easy slicing
    noisy_4d = noisy_volume_3d_denormalized.reshape(
        noisy_volume_3d_denormalized.shape[0],
        noisy_volume_3d_denormalized.shape[1],
        original_a_dim,
        original_e_dim
    )
    target_4d = target_volume_3d_denormalized.reshape(
        target_volume_3d_denormalized.shape[0],
        target_volume_3d_denormalized.shape[1],
        original_a_dim,
        original_e_dim
    )

    # Extract 2D slices
    noisy_2d = noisy_4d[m_idx, e_idx, :, :].cpu().numpy()
    target_2d = target_4d[m_idx, e_idx, :, :].cpu().numpy()
    denoised_2d = denoised_volume_4d[m_idx, e_idx, :, :].cpu().numpy()

    extent = (angleRange[0], angleRange[1], energyRange[0], energyRange[1])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'2D Histograms Comparison for M={m_idx}, E={e_idx}', fontsize=16)

    # Plot Noisy 1
    im1 = axes[0].imshow(noisy_2d, cmap='viridis',extent = extent, aspect='auto')
    axes[0].set_title('Initial Noisy (Noisy1)')
    axes[0].set_xlabel('e-dimension')
    axes[0].set_ylabel('a-dimension')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot Target (Noisy 2)
    im2 = axes[1].imshow(target_2d, cmap='viridis', extent=extent, aspect='auto')
    axes[1].set_title('Target (Noisy2)')
    axes[1].set_xlabel('e-dimension')
    axes[1].set_ylabel('a-dimension')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot Denoised
    im3 = axes[2].imshow(denoised_2d, cmap='viridis', extent=extent, aspect='auto')
    axes[2].set_title('Denoised Output')
    axes[2].set_xlabel('e-dimension')
    axes[2].set_ylabel('a-dimension')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig(f'2d_histograms_M{m_idx}_E{e_idx}.pdf')
    plt.close()


# --- Main Execution Block (for demonstration) ---
if __name__ == "__main__":
    # Ensure NPZ file exists
    if not os.path.exists("./DenoisingDataTransSheet50.npz"):
        raise FileNotFoundError("NPZ file not found. Please ensure 'DenoisingDataTransSheet50.npz' exists.")

    # Load data (this also sets global DATA_MIN_VAL, DATA_MAX_VAL, ORIGINAL_A_DIM, ORIGINAL_E_DIM)
    noisy_volume, target_volume, mask_volume, coordinate_volume, original_a_dim, original_e_dim = \
        load_and_prepare_data(npz_path="./DenoisingDataTransSheet50.npz", required_patch_size=PATCH_SIZE)

    # Initialize dataset for training
    num_total_patches = 20000 # Increased number of patches per epoch
    denoising_dataset = DenoisingDataset(
        noisy_volume=noisy_volume,
        target_volume=target_volume,
        mask_volume=mask_volume,
        coordinate_volume=coordinate_volume,
        patch_size=PATCH_SIZE,
        num_patches=num_total_patches
    )

    train_size = int(0.8 * len(denoising_dataset))
    val_size = len(denoising_dataset) - train_size
    train_dataset, val_dataset = random_split(denoising_dataset, [train_size, val_size])

    print(f"Training dataset size (patches per epoch): {len(train_dataset)}")
    print(f"Validation dataset size (patches per epoch): {len(val_dataset)}")

    batch_size = 10
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model, Loss, and Optimizer - IMPORTANT: in_channels is now 4 (1 data + 3 coordinates)
    model = Noise2Noise3DUNet(in_channels=4, base_channels=16)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005) # Lowered learning rate again
    print(model)
    
    print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    num_epochs = 10
    train_noise2noise(model, train_loader, val_loader, num_epochs, lr=0.00005)

    # --- Denoising Demonstration ---
    print("\n--- Starting Denoising Demonstration ---")
    
    # Ensure the model is loaded or trained before calling denoise_volume
    # For a real application, you would load pre-trained weights here:
    # model.load_state_dict(torch.load('path_to_your_model.pth'))

    # Example call to denoise_volume
    print(f"Denoising noisy volume of shape: {noisy_volume.shape}")
    print(f"Using patch size: {PATCH_SIZE}, overlap ratio: 0.5, batch size: 4")
    
    # Now only receives the 4D volume
    denoised_output_volume_4d = denoise_volume(
        model=model,
        noisy_volume=noisy_volume, # This is the normalized noisy volume
        coordinate_volume=coordinate_volume,
        patch_size=PATCH_SIZE,
        overlap_ratio=0.5, # 50% overlap
        batch_size=4,      # Can adjust this based on GPU memory
        device=device
    )

    if denoised_output_volume_4d is not None:
        print(f"Denoising complete. Output 4D volume shape: {denoised_output_volume_4d.shape}")
        print(f"Denoised 4D volume min/max: {denoised_output_volume_4d.min():.4f}/{denoised_output_volume_4d.max():.4f}")

        # --- Plotting Demonstration ---
        # Generate 10 evenly spaced indices for M and E dimensions
        plot_m_indices = np.linspace(0, noisy_volume.shape[0] - 1, 5, dtype=int).tolist()
        plot_m_indices.append(48, 47)
        plot_e_indices = np.linspace(0, noisy_volume.shape[1] - 1, 5, dtype=int).tolist()
        plot_e_indices.append(48, 47)

        # Iterate through each M and E index to plot individual histograms
        for m_idx in plot_m_indices:
            for e_idx in plot_e_indices:
                # Check if chosen indices are valid before plotting
                if noisy_volume.shape[0] > m_idx and noisy_volume.shape[1] > e_idx:
                    plot_2d_histograms(
                        noisy_volume_3d=noisy_volume,
                        target_volume_3d=target_volume,
                        denoised_volume_4d=denoised_output_volume_4d,
                        m_idx=m_idx,  # Pass individual index
                        e_idx=e_idx,  # Pass individual index
                        original_a_dim=original_a_dim,
                        original_e_dim=original_e_dim,
                        data_min_val=DATA_MIN_VAL,
                        data_max_val=DATA_MAX_VAL
                    )
                else:
                    print(f"Skipping plot for M={m_idx}, E={e_idx} as indices are out of bounds.")
    else:
        print("Denoising failed or 4D reshape was not possible, so plotting cannot proceed.")