import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import random
from tqdm import tqdm
import torch.optim as optim
import os
import matplotlib.pyplot as plt

# Global variables
PATCH_SIZE = (6, 6, 128) # Patch size for training (example: non-cubic)
ORIGINAL_A_DIM = None     # Will store original 'a' dimension from 5D data
ORIGINAL_E_DIM = None     # Will store original 'e' dimension from 5D data

# Squeeze-and-Excite block (remains the same)# Squeeze-and-Excite block
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
    
# Total variation loss (remains the same)
def total_variation_loss(x):
    """Computes total variation loss in 3D"""
    tv_z = torch.mean(torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]))
    tv_y = torch.mean(torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]))
    tv_x = torch.mean(torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]))
    return tv_z + tv_y + tv_x

# load_and_prepare_data function (remains the same)
def load_and_prepare_data(npz_path="./DenoisingDataTransSheet10.npz", required_patch_size=(6, 6, 6)):
    """
    Loads 5D data from an NPZ file, flattens it to 3D volumes,
    generates a mask, and *creates a 3-channel coordinate volume* for positional encoding.

    Args:
        npz_path (str): Path to the .npz file containing the 'histograms' data.
        required_patch_size (tuple): The minimum required patch size (D, H, W)
                                     for the U-Net architecture.

    Returns:
        tuple: (noisy_volume_3d, target_volume_3d, mask_volume_3d, coordinate_volume_3d, original_a, original_e)
               as PyTorch tensors and integers.
    """
    global ORIGINAL_A_DIM, ORIGINAL_E_DIM # Declare intent to modify global variables

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

        noisydata1_4d = histograms_np[0] # Shape: [M, E, a, e] (Input to model)
        noisydata2_4d = histograms_np[1] # Shape: [M, E, a, e] (Target/clean histogram)

        M_dim, E_dim, a_dim, e_dim = noisydata1_4d.shape
        ORIGINAL_A_DIM = a_dim
        ORIGINAL_E_DIM = e_dim

        # Flatten 'a' and 'e' dimensions into a single 'W' dimension
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

# DenoisingDataset class (remains the same)
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
        self.coordinate_volume = coordinate_volume
        self.patch_D, self.patch_H, self.patch_W = patch_size
        self.volume_D, self.volume_H, self.volume_W = noisy_volume.shape
        self.num_patches = num_patches

        self.valid_non_zero_centers = self._get_valid_non_zero_centers()
        print(f"Found {len(self.valid_non_zero_centers)} valid non-zero centers for patch extraction.")

        self.patch_coords = self._generate_patch_coordinates()
        print(f"Generated {len(self.patch_coords)} patch coordinates.")

    def _get_valid_non_zero_centers(self):
        non_zero_indices = torch.nonzero(self.mask_volume, as_tuple=False)
        valid_centers = [idx.tolist() for idx in tqdm(non_zero_indices, desc="Collecting all non-zero centers")]
        return valid_centers

    def _generate_patch_coordinates(self):
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
        d = random.randint(0, self.volume_D - 1)
        h = random.randint(0, self.volume_H - 1)
        w = random.randint(0, self.volume_W - 1)
        return (d, h, w)

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
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
        coordinate_patch_slice = self.coordinate_volume[
            :, # All 3 channels
            slice_start_d:slice_end_d,
            slice_start_h:slice_end_h,
            slice_start_w:slice_end_w
        ]

        # Use the actual dimensions of the slice, not the requested patch size,
        # when determining if 'reflect' is safe.
        current_d_dim = noisy_patch_slice.shape[0]
        current_h_dim = noisy_patch_slice.shape[1]
        current_w_dim = noisy_patch_slice.shape[2]

        padding = (pad_left_w, pad_right_w,
                   pad_left_h, pad_right_h,
                   pad_left_d, pad_right_d)

        # For probability data, always pad with 0
        pad_mode = 'constant'
        pad_value = 0.0

        noisy_patch_slice_ch = noisy_patch_slice.unsqueeze(0)
        target_patch_slice_ch = target_patch_slice.unsqueeze(0)

        noisy_patch = F.pad(noisy_patch_slice_ch, padding, mode=pad_mode, value=pad_value)
        target_patch = F.pad(target_patch_slice_ch, padding, mode=pad_mode, value=pad_value)

        # Mask and coordinate patches should always be padded with 0
        mask_patch = F.pad(mask_patch_slice.float(), padding, mode='constant', value=0).bool()
        coordinate_patch = F.pad(coordinate_patch_slice, padding, mode='constant', value=0)


        # --- Data Augmentation: Random 3D Flips ---
        # Apply random flips along D, H, and W axes
        if random.random() < 0.5: # 50% chance to flip along D
            noisy_patch = torch.flip(noisy_patch, [1]) # Dim 1 is D for (C, D, H, W)
            target_patch = torch.flip(target_patch, [1])
            mask_patch = torch.flip(mask_patch, [0]) # Dim 0 is D for (D, H, W)
            coordinate_patch = torch.flip(coordinate_patch, [1])
        if random.random() < 0.5: # 50% chance to flip along H
            noisy_patch = torch.flip(noisy_patch, [2]) # Dim 2 is H for (C, D, H, W)
            target_patch = torch.flip(target_patch, [2])
            mask_patch = torch.flip(mask_patch, [1]) # Dim 1 is H for (D, H, W)
            coordinate_patch = torch.flip(coordinate_patch, [2])
        if random.random() < 0.5: # 50% chance to flip along W
            noisy_patch = torch.flip(noisy_patch, [3]) # Dim 3 is W for (C, D, H, W)
            target_patch = torch.flip(target_patch, [3])
            mask_patch = torch.flip(mask_patch, [2]) # Dim 2 is W for (D, H, W)
            coordinate_patch = torch.flip(coordinate_patch, [3])

        # --- Data Augmentation: Random 3D Rotations (Targeted) ---
        # Apply rotations ONLY if the relevant dimensions are equal
        # For (C, D, H, W) tensors: D=1, H=2, W=3
        # For (D, H, W) tensors (mask): D=0, H=1, W=2

        # Rotate around W-axis if D and H dimensions are equal (dims=(1,2) for 4D, (0,1) for 3D)
        if self.patch_D == self.patch_H and random.random() < 0.5:
            k_rotations = random.randint(1, 3) # Rotate 90, 180, or 270 degrees
            noisy_patch = torch.rot90(noisy_patch, k=k_rotations, dims=(1, 2))
            target_patch = torch.rot90(target_patch, k=k_rotations, dims=(1, 2))
            mask_patch = torch.rot90(mask_patch, k=k_rotations, dims=(0, 1))
            coordinate_patch = torch.rot90(coordinate_patch, k=k_rotations, dims=(1, 2))

        # Rotate around H-axis if D and W dimensions are equal (dims=(1,3) for 4D, (0,2) for 3D)
        if self.patch_D == self.patch_W and random.random() < 0.5:
            k_rotations = random.randint(1, 3)
            noisy_patch = torch.rot90(noisy_patch, k=k_rotations, dims=(1, 3))
            target_patch = torch.rot90(target_patch, k=k_rotations, dims=(1, 3))
            mask_patch = torch.rot90(mask_patch, k=k_rotations, dims=(0, 2))
            coordinate_patch = torch.rot90(coordinate_patch, k=k_rotations, dims=(1, 3))

        # Rotate around D-axis if H and W dimensions are equal (dims=(2,3) for 4D, (1,2) for 3D)
        if self.patch_H == self.patch_W and random.random() < 0.5:
            k_rotations = random.randint(1, 3)
            noisy_patch = torch.rot90(noisy_patch, k=k_rotations, dims=(2, 3))
            target_patch = torch.rot90(target_patch, k=k_rotations, dims=(2, 3))
            mask_patch = torch.rot90(mask_patch, k=k_rotations, dims=(1, 2))
            coordinate_patch = torch.rot90(coordinate_patch, k=k_rotations, dims=(2, 3))

        # Assertions to ensure the patches have the correct final size
        assert noisy_patch.shape == (1, self.patch_D, self.patch_H, self.patch_W), \
            f"Noisy patch shape mismatch after rotation: Expected (1, {self.patch_D}, {self.patch_H}, {self.patch_W}), Got {noisy_patch.shape}"
        assert target_patch.shape == (1, self.patch_D, self.patch_H, self.patch_W), \
            f"Target patch shape mismatch after rotation: Expected (1, {self.patch_D}, {self.patch_H}, {self.patch_W}), Got {target_patch.shape}"
        assert mask_patch.shape == (self.patch_D, self.patch_H, self.patch_W), \
            f"Mask patch shape mismatch after rotation: Expected ({self.patch_D}, {self.patch_H}, {self.patch_W}), Got {mask_patch.shape}"
        assert coordinate_patch.shape == (3, self.patch_D, self.patch_H, self.patch_W), \
            f"Coord patch shape mismatch after rotation: Expected (3, {self.patch_D}, {self.patch_H}, {self.patch_W}), Got {coordinate_patch.shape}"

        return noisy_patch, target_patch, mask_patch, coordinate_patch

# denoise_volume function (modified to directly use model output)
def denoise_volume(
    model,
    noisy_full_volume,
    patch_size,
    device='cuda',
    overlap_factor=0.5,
    batch_size=4
):
    """
    Denoises a full 3D volume by processing it in overlapping patches using the trained model.
    The model is assumed to directly output the denoised volume.
    Handles coordinate volume generation and reconstruction.

    Args:
        model (nn.Module): The trained Noise2Noise3DUNet model.
        noisy_full_volume (torch.Tensor): The full 3D noisy data volume (D, H, W). Expected values [0, 1].
        original_a (int): Original 'a' dimension from 5D data, for reshaping.
        original_e (int): Original 'e' dimension from 5D data, for reshaping.
        patch_size (tuple): The (patch_D, patch_H, patch_W) size expected by the model.
        device (str): Device to run inference on ('cuda' or 'cpu').
        overlap_factor (float): Fraction of overlap between patches (e.g., 0.5 for 50%).
        batch_size (int): Number of patches to process simultaneously during inference.

    Returns:
        torch.Tensor: The denoised 3D volume, with values clamped to [0, 1].
    """
    model.eval()
    model = model.to(device)

    full_D, full_H, full_W = noisy_full_volume.shape
    patch_D, patch_H, patch_W = patch_size

    # Calculate step sizes based on overlap
    step_D = int(patch_D * (1 - overlap_factor))
    step_H = int(patch_H * (1 - overlap_factor))
    step_W = int(patch_W * (1 - overlap_factor))

    # Ensure step sizes are at least 1
    step_D = max(1, step_D)
    step_H = max(1, step_H)
    step_W = max(1, step_W)

    denoised_full_volume_accumulator = torch.zeros_like(noisy_full_volume, dtype=torch.float32)
    overlap_counter = torch.zeros_like(noisy_full_volume, dtype=torch.float32)

    # Generate the coordinate volume for the full input
    coords_d = torch.linspace(0, 1, full_D).reshape(full_D, 1, 1).expand(full_D, full_H, full_W)
    coords_h = torch.linspace(0, 1, full_H).reshape(1, full_H, 1).expand(full_D, full_H, full_W)
    coords_w = torch.linspace(0, 1, full_W).reshape(1, 1, full_W).expand(full_D, full_H, full_W)
    full_coordinate_volume = torch.stack((coords_d, coords_h, coords_w), dim=0).float() # (3, D, H, W)

    # Collect patches for batched inference
    input_patches_list = []
    patch_coords_list = [] # Store coordinates to place denoised patches back

    for d in range(0, full_D, step_D):
        for h in range(0, full_H, step_H):
            for w in range(0, full_W, step_W):
                end_d = min(d + patch_D, full_D)
                end_h = min(h + patch_H, full_H)
                end_w = min(w + patch_W, full_W)

                start_d = end_d - patch_D
                start_h = end_h - patch_H
                start_w = end_w - patch_W

                if start_d < 0: start_d = 0
                if start_h < 0: start_h = 0
                if start_w < 0: start_w = 0

                noisy_sub_volume = noisy_full_volume[start_d:end_d, start_h:end_h, start_w:end_w]
                coord_sub_volume = full_coordinate_volume[:, start_d:end_d, start_h:end_h, start_w:end_w]

                current_patch_D, current_patch_H, current_patch_W = noisy_sub_volume.shape
                
                pad_left_d = 0
                pad_right_d = patch_D - current_patch_D
                pad_left_h = 0
                pad_right_h = patch_H - current_patch_H
                pad_left_w = 0
                pad_right_w = patch_W - current_patch_W

                padding_tuple_for_Fpad = (pad_left_w, pad_right_w,
                                          pad_left_h, pad_right_h,
                                          pad_left_d, pad_right_d)

                padded_noisy_patch = F.pad(noisy_sub_volume.unsqueeze(0).unsqueeze(0),
                                           padding_tuple_for_Fpad, mode='constant', value=0.0)
                padded_coord_patch = F.pad(coord_sub_volume.unsqueeze(0),
                                           padding_tuple_for_Fpad, mode='constant', value=0.0)

                input_patches_list.append(torch.cat([padded_noisy_patch, padded_coord_patch], dim=1))
                patch_coords_list.append((start_d, end_d, start_h, end_h, start_w, end_w, current_patch_D, current_patch_H, current_patch_W))

    # Process patches in batches
    num_batches = (len(input_patches_list) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Denoising Patches"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(input_patches_list))
            
            batch_input = torch.cat(input_patches_list[batch_start:batch_end], dim=0).to(device)

            # Model directly predicts the denoised output
            denoised_batch_output = model(batch_input) # (B, 1, D, H, W)

            denoised_batch_output = denoised_batch_output.cpu() # Move to CPU for reconstruction

            # Reconstruct the full volume
            for j, patch_output in enumerate(denoised_batch_output):
                (start_d, end_d, start_h, end_h, start_w, end_w, current_patch_D, current_patch_H, current_patch_W) = patch_coords_list[batch_start + j]

                # Unpad the denoised patch to its original (non-padded) size for reconstruction
                unpadded_denoised_patch = patch_output[
                    :,
                    :current_patch_D,
                    :current_patch_H,
                    :current_patch_W
                ].squeeze(0) # Remove channel dim (1)

                denoised_full_volume_accumulator[start_d:end_d, start_h:end_h, start_w:end_w] += unpadded_denoised_patch
                overlap_counter[start_d:end_d, start_h:end_h, start_w:end_w] += 1.0

    # Average the overlapping regions
    overlap_counter[overlap_counter == 0] = 1.0 # Avoid division by zero for uncovered regions
    denoised_full_volume_accumulator /= overlap_counter

    # The model's final sigmoid should already ensure [0,1], but a final clamp
    # acts as a safeguard against floating point inaccuracies if any.
    denoised_full_volume_accumulator = torch.clamp(denoised_full_volume_accumulator, min=0.0, max=1.0)
    print("Denoised volume clamped to [0.0, 1.0] for probability integrity.")

    return denoised_full_volume_accumulator

# train_noise2noise function (modified for direct output training)
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
            # noisy_patch and target_patch are already (B, 1, D, H, W) from __getitem__ + DataLoader
            noisy_patch = noisy_patch.to(device)
            target_patch = target_patch.to(device) # This is now the direct target for the model
            coord_patch = coord_patch.to(device) # (B, 3, D, H, W)
            input_patch = torch.cat([noisy_patch, coord_patch], dim=1) # Concatenates to (B, 4, D, H, W)

            optimizer.zero_grad()
            
            # Model outputs the predicted clean histogram
            predicted_clean_histogram = model(input_patch)

            # The target for the model is directly the target_patch
            loss = mse_loss(predicted_clean_histogram, target_patch)

            # Add optional L1
            if "l1" in loss_type:
                # L1 loss is directly on the difference between prediction and target
                loss += 0.1 * l1_loss(predicted_clean_histogram, target_patch)

            # Add optional Total Variation (TV) - applied to the predicted clean histogram for smoothness
            if "tv" in loss_type:
                loss += 0.01 * total_variation_loss(predicted_clean_histogram)

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
                noisy_patch = noisy_patch.to(device)
                target_patch = target_patch.to(device)
                coord_patch = coord_patch.to(device)
                input_patch = torch.cat([noisy_patch, coord_patch], dim=1)

                predicted_clean_histogram = model(input_patch)
                
                loss = mse_loss(predicted_clean_histogram, target_patch)

                if "l1" in loss_type:
                    loss += 0.1 * l1_loss(predicted_clean_histogram, target_patch)
                if "tv" in loss_type:
                    loss += 0.01 * total_variation_loss(predicted_clean_histogram)

                val_loss += loss.item()
                avg_val_batch_loss = val_loss / (batch_idx + 1)
                val_bar.set_postfix(loss=avg_val_batch_loss)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.6f}")

        # Step scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.2e}")

# plot_2d_histograms function (remains the same)
def plot_2d_histograms(
    noisy_volume_3d: torch.Tensor,
    target_volume_3d: torch.Tensor,
    denoised_volume_3d: torch.Tensor, # Changed from 4D to 3D input
    m_idx: int,
    e_idx: int,
    original_a_dim: int,
    original_e_dim: int,
    angleRange = (0,70),
    energyRange = (-0.6, 0)
):
    """
    Plots 2D histograms comparing initial noisy, target (noisy2), and denoised data
    for a specific M and E index, after reshaping 3D volumes back to 4D.

    Args:
        noisy_volume_3d (torch.Tensor): The full 3D noisy data volume (M, E, a*e).
        target_volume_3d (torch.Tensor): The full 3D target data volume (M, E, a*e).
        denoised_volume_3d (torch.Tensor): The full 3D denoised volume (M, E, a*e).
        m_idx (int): The index for the M dimension (depth/first dimension).
        e_idx (int): The index for the E dimension (height/second dimension).
        original_a_dim (int): The original 'a' dimension from the 5D data.
        original_e_dim (int): The original 'e' dimension from the 5D data.
        angleRange (tuple): Range for the 'a' dimension (x-axis for plots).
        energyRange (tuple): Range for the 'e' dimension (y-axis for plots).
    """
    # Ensure indices are within bounds
    if not (0 <= m_idx < noisy_volume_3d.shape[0] and
            0 <= e_idx < noisy_volume_3d.shape[1]):
        print(f"Error: M index ({m_idx}) or E index ({e_idx}) out of bounds for volumes of shape {noisy_volume_3d.shape[:2]}.")
        return

    # Reshape 3D volumes to 4D (M, E, a, e) for plotting
    M_dim_full, E_dim_full, _ = noisy_volume_3d.shape
    
    noisy_volume_4d_reshaped = noisy_volume_3d.reshape(M_dim_full, E_dim_full, original_a_dim, original_e_dim)
    target_volume_4d_reshaped = target_volume_3d.reshape(M_dim_full, E_dim_full, original_a_dim, original_e_dim)
    denoised_volume_4d_reshaped = denoised_volume_3d.reshape(M_dim_full, E_dim_full, original_a_dim, original_e_dim)


    # Extract 2D slices for the given m_idx and e_idx
    noisy_2d = noisy_volume_4d_reshaped[m_idx, e_idx, :, :].cpu().numpy()
    target_2d = target_volume_4d_reshaped[m_idx, e_idx, :, :].cpu().numpy()
    denoised_2d = denoised_volume_4d_reshaped[m_idx, e_idx, :, :].cpu().numpy()

    extent = (angleRange[0], angleRange[1], energyRange[0], energyRange[1]) 

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'2D Histograms Comparison for M={m_idx}, E={e_idx}', fontsize=16)

    # Plot Noisy 1
    im1 = axes[0].imshow(noisy_2d, cmap='viridis', extent=extent, aspect='auto', origin='lower')
    axes[0].set_title('Initial Noisy (Noisy1)')
    axes[0].set_xlabel('Energy') # Assuming e-dimension corresponds to energy
    axes[0].set_ylabel('Angle') # Assuming a-dimension corresponds to angle
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot Target (Noisy 2)
    im2 = axes[1].imshow(target_2d, cmap='viridis', extent=extent, aspect='auto', origin='lower')
    axes[1].set_title('Target (Noisy2)')
    axes[1].set_xlabel('Energy')
    axes[1].set_ylabel('Angle')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot Denoised
    im3 = axes[2].imshow(denoised_2d, cmap='viridis', extent=extent, aspect='auto', origin='lower')
    axes[2].set_title('Denoised Output')
    axes[2].set_xlabel('Energy')
    axes[2].set_ylabel('Angle')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig(f'2d_histograms_M{m_idx}_E{e_idx}.pdf')
    plt.close()


# --- Main Execution Block ---
if __name__ == "__main__":
    # Ensure NPZ file exists
    if not os.path.exists("./DenoisingDataTransSheet50.npz"):
        raise FileNotFoundError("NPZ file not found. Please ensure 'DenoisingDataTransSheet50.npz' exists.")

    # Load data (this also sets global ORIGINAL_A_DIM, ORIGINAL_E_DIM)
    noisy_volume_3d, target_volume_3d, mask_volume, coordinate_volume, original_a_dim, original_e_dim = \
        load_and_prepare_data(npz_path="./DenoisingDataTransSheet50.npz", required_patch_size=PATCH_SIZE)

    # Initialize dataset for training
    num_total_patches = 10000 # Increased number of patches per epoch
    denoising_dataset = DenoisingDataset(
        noisy_volume=noisy_volume_3d,
        target_volume=target_volume_3d,
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

    batch_size = 12
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model, Loss, and Optimizer - IMPORTANT: in_channels is now 4 (1 data + 3 coordinates)
    model = Noise2Noise3DUNet(in_channels=4, base_channels=16)
    # The criterion is implicitly handled within train_noise2noise for MSE, L1, TV
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005) # Lowered learning rate again
    print(model)
    
    print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    num_epochs = 10
    train_noise2noise(model, train_loader, val_loader, num_epochs, lr=0.00001)

    # --- Denoising Demonstration ---
    print("\n--- Starting Denoising Demonstration ---")
    
    # Ensure the model is loaded or trained before calling denoise_volume
    # For a real application, you would load pre-trained weights here:
    # model.load_state_dict(torch.load('path_to_your_model.pth'))

    # Example call to denoise_volume
    print(f"Denoising noisy volume of shape: {noisy_volume_3d.shape}")
    print(f"Using patch size: {PATCH_SIZE}, overlap factor: 0.5, batch size: 4")
    
    # Call denoise_volume
    denoised_output_volume_3d = denoise_volume(
        model=model,
        noisy_full_volume=noisy_volume_3d, # Corrected parameter name
        patch_size=PATCH_SIZE,
        overlap_factor=0.5, # Corrected parameter name
        batch_size=4,       # Can adjust this based on GPU memory
        device=device
    )

    if denoised_output_volume_3d is not None:
        print(f"Denoising complete. Output 3D volume shape: {denoised_output_volume_3d.shape}")
        print(f"Denoised 3D volume min/max: {denoised_output_volume_3d.min():.4f}/{denoised_output_volume_3d.max():.4f}")

        # Reshape the 3D denoised volume back to 4D (M, E, a, e) for plotting
        # Assuming the first two dimensions of noisy_volume_3d are M and E
        M_dim_full, E_dim_full, _ = noisy_volume_3d.shape
        denoised_output_volume_4d = denoised_output_volume_3d.reshape(M_dim_full, E_dim_full, original_a_dim, original_e_dim)
        
        print(f"Reshaped denoised output to 4D shape: {denoised_output_volume_4d.shape}")

        # --- Plotting Demonstration ---
        # Generate 10 evenly spaced indices for M and E dimensions
        plot_m_indices = np.linspace(0, noisy_volume_3d.shape[0] - 1, 5, dtype=int).tolist()
        plot_m_indices.extend([48, 47])
        plot_e_indices = np.linspace(0, noisy_volume_3d.shape[1] - 1, 5, dtype=int).tolist()
        plot_e_indices.extend([48, 47])

        # Iterate through each M and E index to plot individual histograms
        for m_idx in plot_m_indices:
            for e_idx in plot_e_indices:
                # Check if chosen indices are valid before plotting
                if noisy_volume_3d.shape[0] > m_idx and noisy_volume_3d.shape[1] > e_idx:
                    plot_2d_histograms(
                        noisy_volume_3d=noisy_volume_3d, # Pass 3D noisy volume
                        target_volume_3d=target_volume_3d, # Pass 3D target volume
                        denoised_volume_3d=denoised_output_volume_3d, # Pass 3D denoised volume
                        m_idx=m_idx,
                        e_idx=e_idx,
                        original_a_dim=original_a_dim,
                        original_e_dim=original_e_dim,
                    )
                else:
                    print(f"Skipping plot for M={m_idx}, E={e_idx} as indices are out of bounds.")
    else:
        print("Denoising failed or 4D reshape was not possible, so plotting cannot proceed.")