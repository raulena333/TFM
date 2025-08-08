import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
from tqdm import tqdm # For progress bars
import matplotlib.pyplot as plt # For plotting

# --- 1. Data Loading Function (Original - for reference, not used in main for synthetic data) ---
def load_and_prepare_data(npz_path="./DenoisingDataTransSheet10.npz"):
    """
    Loads 5D data from an NPZ file, keeping it as (M, E, a, e) for dataset.
    Generates a mask and a 4-channel coordinate volume for positional encoding.

    Args:
        npz_path (str): Path to the .npz file containing the 'histograms' data.

    Returns:
        tuple: (noisy_h1, noisy_h2, mask_volume_4d, coordinate_volume_4d,
                original_M, original_E, original_a, original_e)
                as PyTorch tensors and integers.
    """
    print(f"Loading data from: {npz_path}")
    try:
        loaded_data = np.load(np.npz_path)
        histograms_np = loaded_data['histograms'].astype(np.float32)
        print(f"Loaded 'histograms' data with shape: {histograms_np.shape}")

        if histograms_np.ndim != 5 or histograms_np.shape[0] != 2:
            raise ValueError(
                "Expected 'histograms' data to be 5D with shape [2, M, E, a, e]. "
                f"Got shape: {histograms_np.shape}"
            )

        noisy_h1 = histograms_np[0] # Shape: [M, E, a, e] (Input to model)
        noisy_h2 = histograms_np[1] # Shape: [M, E, a, e] (Target/clean histogram)

        original_M, original_E, original_a, original_e = noisy_h1.shape

        # Create mask based on non-zero values in the first noisy data
        mask_volume_4d = (noisy_h1 != 0)

        # --- Generate Positional Encoding (Coordinate Volume) ---
        # Create normalized coordinate grids for M, E, a, e dimensions
        coords_m = torch.linspace(0, 1, original_M).reshape(original_M, 1, 1, 1).expand(original_M, original_E, original_a, original_e)
        coords_e = torch.linspace(0, 1, original_E).reshape(1, original_E, 1, 1).expand(original_M, original_E, original_a, original_e)
        coords_a = torch.linspace(0, 1, original_a).reshape(1, 1, original_a, 1).expand(original_M, original_E, original_a, original_e)
        coords_e_idx = torch.linspace(0, 1, original_e).reshape(1, 1, 1, original_e).expand(original_M, original_E, original_a, original_e)

        # Stack them to create a (4, M, E, a, e) coordinate volume
        coordinate_volume_4d = torch.stack((coords_m, coords_e, coords_a, coords_e_idx), dim=0).float()
        print(f"Coordinate Volume Shape: {coordinate_volume_4d.shape}")


        print(f"Noisy Volume (4D) Shape: {noisy_h1.shape}")
        print(f"Target Volume (4D) Shape: {noisy_h2.shape}")
        print(f"Mask Volume (4D) Shape: {mask_volume_4d.shape} (derived from non-zero voxels)")

        # Convert NumPy arrays to PyTorch tensors
        return torch.from_numpy(noisy_h1), torch.from_numpy(noisy_h2), \
               torch.from_numpy(mask_volume_4d), coordinate_volume_4d, \
               original_M, original_E, original_a, original_e

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {npz_path}. Please ensure the file exists in the correct directory.")
    except Exception as e:
        raise Exception(f"Error loading data from {npz_path}: {e}")

# --- New: Synthetic Data Generation Function ---
def generate_synthetic_noisy_data(M_dim, E_dim, A_dim, e_dim, noise_level=0.05):
    """
    Generates synthetic 4D noisy (noisy_h1, noisy_h2) and clean histogram data for testing.
    The clean data will have a square-like peak in (a,e) space that shifts based on M and E indices,
    and noise will only be present within these non-zero regions.

    Args:
        M_dim (int): Dimension for Material.
        E_dim (int): Dimension for Initial Energy.
        A_dim (int): Dimension for Scattered Angle.
        e_dim (int): Dimension for Final Energy.
        noise_level (float): Standard deviation of Gaussian noise to add.

    Returns:
        tuple: (noisy_h1, noisy_h2_input_to_model, clean_target_data, mask_volume_4d, coordinate_volume_4d,
                original_M, original_E, original_a, original_e)
                as PyTorch tensors and integers.
    """
    print(f"Generating synthetic data with dimensions: M={M_dim}, E={E_dim}, A={A_dim}, e={e_dim}")

    clean_target_data = torch.zeros((M_dim, E_dim, A_dim, e_dim), dtype=torch.float32)

    # Create coordinate grids for (a,e) for square peak generation
    a_grid = torch.arange(A_dim).float()
    e_grid = torch.arange(e_dim).float()
    
    # Create 2D meshgrid for (a,e)
    a_mesh, e_mesh = torch.meshgrid(a_grid, e_grid, indexing='ij')

    # Iterate through M and E dimensions to place shifting square peaks
    for m_idx in range(M_dim):
        for e_idx in range(E_dim):
            # Normalize M and E indices to [0, 1] range
            norm_m = m_idx / (M_dim - 1) if M_dim > 1 else 0.5
            norm_e = e_idx / (E_dim - 1) if E_dim > 1 else 0.5

            # Determine peak center (mean) in (a,e) space based on normalized M and E
            # Peak moves right with M, and up with E
            center_a = A_dim * (0.1 + 0.8 * norm_m)
            center_e = e_dim * (0.1 + 0.8 * norm_e)

            # Base size of the square
            base_square_size_a = max(1, A_dim // 20) # Minimum size 1, or 1/20 of dim
            base_square_size_e = max(1, e_dim // 20)

            # Make size dependent on M, E slightly (e.g., smaller for higher M, E)
            current_square_size_a = max(2, int(base_square_size_a * (1.5 - 0.5 * norm_m)))
            current_square_size_e = max(2, int(base_square_size_e * (1.5 - 0.5 * norm_e)))

            # Ensure sizes are even for symmetry around center, or adjust center
            current_square_size_a = current_square_size_a if current_square_size_a % 2 == 0 else current_square_size_a + 1
            current_square_size_e = current_square_size_e if current_square_size_e % 2 == 0 else current_square_size_e + 1

            # Calculate coordinates relative to the center of the current (a,e) grid
            rel_a = a_mesh - center_a
            rel_e = e_mesh - center_e

            # Simplified smooth square (using sigmoid-like falloff)
            # This creates a more 'square' shape with fading edges
            # The 'sharpness' parameter controls how quickly it fades
            sharpness = 4.0 # Higher value means sharper edges
            
            # Create a 2D grid for the square
            square_a = 1.0 / (1.0 + torch.exp(sharpness * (torch.abs(rel_a) - current_square_size_a / 2.0)))
            square_e = 1.0 / (1.0 + torch.exp(sharpness * (torch.abs(rel_e) - current_square_size_e / 2.0)))
            
            soft_square = square_a * square_e # Product creates the square shape
            
            # Scale the peak intensity (slightly increased for better visibility)
            peak_intensity = 1.0 + 0.5 * (norm_m + norm_e) / 2 # Increased base intensity
            current_slice = soft_square * peak_intensity

            # Normalize the current (a,e) slice such that its sum is 1
            slice_sum = current_slice.sum()
            if slice_sum > 0:
                current_slice /= slice_sum
            
            clean_target_data[m_idx, e_idx, :, :] = current_slice

    # Create a mask for where the clean data is non-zero (or above a very small threshold)
    # This mask will determine where noise is applied
    non_zero_mask = (clean_target_data > 1e-6).float() # Use a small threshold to avoid floating point issues

    # Generate noisy_h1 and noisy_h2
    noise1 = torch.randn_like(clean_target_data) * noise_level
    noise2 = torch.randn_like(clean_target_data) * noise_level * 1.1

    # Apply noise ONLY where the clean data is non-zero
    noisy_h1 = clean_target_data + (noise1 * non_zero_mask)
    noisy_h2_input_to_model = clean_target_data + (noise2 * non_zero_mask)

    noisy_h1 = torch.clamp(noisy_h1, min=0.0)
    noisy_h2_input_to_model = torch.clamp(noisy_h2_input_to_model, min=0.0)

    # --- Normalize noisy_h1 and noisy_h2 per (M, E) slice ---
    for m_idx in range(M_dim):
        for e_idx in range(E_dim):
            # Normalize noisy_h1 slice
            slice_sum_h1 = noisy_h1[m_idx, e_idx, :, :].sum()
            if slice_sum_h1 > 0:
                noisy_h1[m_idx, e_idx, :, :] /= slice_sum_h1

            # Normalize noisy_h2 slice
            slice_sum_h2 = noisy_h2_input_to_model[m_idx, e_idx, :, :].sum()
            if slice_sum_h2 > 0:
                noisy_h2_input_to_model[m_idx, e_idx, :, :] /= slice_sum_h2
    # --- End Normalization ---

    original_M, original_E, original_a, original_e = noisy_h1.shape

    # Create mask based on non-zero values in the noisy_h1 (or clean_target_data for "true" mask)
    mask_volume_4d = (noisy_h1 != 0) 

    # --- Generate Positional Encoding (Coordinate Volume) ---
    coords_m = torch.linspace(0, 1, original_M).reshape(original_M, 1, 1, 1).expand(original_M, original_E, original_a, original_e)
    coords_e = torch.linspace(0, 1, original_E).reshape(1, original_E, 1, 1).expand(original_M, original_E, original_a, original_e)
    coords_a = torch.linspace(0, 1, original_a).reshape(1, 1, original_a, 1).expand(original_M, original_E, original_a, original_e)
    coords_e_idx = torch.linspace(0, 1, original_e).reshape(1, 1, 1, original_e).expand(original_M, original_E, original_a, original_e)

    coordinate_volume_4d = torch.stack((coords_m, coords_e, coords_a, coords_e_idx), dim=0).float()
    print(f"Synthetic Coordinate Volume Shape: {coordinate_volume_4d.shape}")
    print(f"Synthetic Noisy Volume 1 (4D) Shape: {noisy_h1.shape}")
    print(f"Synthetic Noisy Volume 2 (4D) Shape: {noisy_h2_input_to_model.shape}")
    print(f"Synthetic Clean Target Volume (4D) Shape: {clean_target_data.shape}")
    print(f"Synthetic Mask Volume (4D) Shape: {mask_volume_4d.shape}")

    return noisy_h1, noisy_h2_input_to_model, clean_target_data, mask_volume_4d, coordinate_volume_4d, \
           original_M, original_E, original_a, original_e

# --- 2. Denoising Dataset Class ---
class DenoisingDataset(Dataset):
    """
    A PyTorch Dataset for extracting 4D patches (sub-volumes of M, E, a, e)
    from noisy, target, mask, and coordinate volumes.

    Implements a mixed sampling strategy: 80% patches centered on non-zero mask
    regions, 20% randomly sampled. Includes random 4D flips for data augmentation.
    """
    def __init__(self, noisy_h1_volume, noisy_h2_volume, target_volume, mask_volume, coordinate_volume, patch_size, num_patches=20000):
        """
        Args:
            noisy_h1_volume (torch.Tensor): The first 4D noisy data volume (M, E, a, e).
            noisy_h2_volume (torch.Tensor): The second 4D noisy data volume (M, E, a, e).
            target_volume (torch.Tensor): The 4D clean target data volume (M, E, a, e).
            mask_volume (torch.Tensor): The 4D boolean mask volume (M, E, a, e), True for non-zero.
            coordinate_volume (torch.Tensor): The 4D coordinate volume (4, M, E, a, e).
            patch_size (tuple): The desired patch size (patch_M, patch_E, patch_a, patch_e).
            num_patches (int): The total number of patches to generate for the dataset.
        """
        self.noisy_h1_volume_full = noisy_h1_volume
        self.noisy_h2_volume_full = noisy_h2_volume
        self.target_volume_full = target_volume
        self.mask_volume_full = mask_volume
        self.coordinate_volume_full = coordinate_volume

        self.original_M, self.original_E, self.original_a, self.original_e = noisy_h1_volume.shape

        self.patch_M, self.patch_E, self.patch_a, self.patch_e = patch_size
        self.num_patches = num_patches

        # Ensure patch size is not larger than original dimensions
        if not (self.patch_M <= self.original_M and
                self.patch_E <= self.original_E and
                self.patch_a <= self.original_a and
                self.patch_e <= self.original_e):
            raise ValueError(f"Patch size {patch_size} must be less than or equal to original volume size "
                             f"({self.original_M, self.original_E, self.original_a, self.original_e})")

        self.valid_non_zero_centers = self._get_valid_non_zero_centers()
        print(f"Found {len(self.valid_non_zero_centers)} valid non-zero centers for patch extraction.")

        self.patch_coords = self._generate_patch_coordinates()
        print(f"Generated {len(self.patch_coords)} patch coordinates.")

    def _get_valid_non_zero_centers(self):
        # Non-zero indices across all 4 dimensions
        non_zero_indices = torch.nonzero(self.mask_volume_full, as_tuple=False)
        valid_centers = []
        for idx in tqdm(non_zero_indices, desc="Collecting non-zero centers"):
            m, e, a, final_e = idx.tolist()
            # A center is valid if a patch of `patch_size` can be extracted around it
            # without going too far out of bounds (considering padding will handle edges)
            # For sampling, we just need a center point. Padding will handle the rest.
            valid_centers.append((m, e, a, final_e))
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
        # Random center for a 4D patch
        m = random.randint(0, self.original_M - 1)
        e = random.randint(0, self.original_E - 1)
        a = random.randint(0, self.original_a - 1)
        final_e = random.randint(0, self.original_e - 1)
        return (m, e, a, final_e)

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
        center_m, center_e, center_a, center_final_e = self.patch_coords[idx]

        # Calculate start and end indices for slicing, considering half patch size
        half_patch_M = self.patch_M // 2
        half_patch_E = self.patch_E // 2
        half_patch_a = self.patch_a // 2
        half_patch_e = self.patch_e // 2

        start_m = center_m - half_patch_M
        end_m = start_m + self.patch_M
        start_e = center_e - half_patch_E
        end_e = start_e + self.patch_E
        start_a = center_a - half_patch_a
        end_a = start_a + self.patch_a
        start_final_e = center_final_e - half_patch_e
        end_final_e = start_final_e + self.patch_e

        # Calculate padding needed for each dimension
        pad_left_m = max(0, -start_m)
        pad_right_m = max(0, end_m - self.original_M)
        pad_left_e = max(0, -start_e)
        pad_right_e = max(0, end_e - self.original_E)
        pad_left_a = max(0, -start_a)
        pad_right_a = max(0, end_a - self.original_a)
        pad_left_final_e = max(0, -start_final_e)
        pad_right_final_e = max(0, end_final_e - self.original_e)

        # Calculate actual slice ranges
        slice_start_m = max(0, start_m)
        slice_end_m = min(self.original_M, end_m)
        slice_start_e = max(0, start_e)
        slice_end_e = min(self.original_E, end_e)
        slice_start_a = max(0, start_a)
        slice_end_a = min(self.original_a, end_a)
        slice_start_final_e = max(0, start_final_e)
        slice_end_final_e = min(self.original_e, end_final_e)

        # Extract 4D slices
        noisy_h1_patch_slice = self.noisy_h1_volume_full[
            slice_start_m:slice_end_m,
            slice_start_e:slice_end_e,
            slice_start_a:slice_end_a,
            slice_start_final_e:slice_end_final_e
        ]
        noisy_h2_patch_slice = self.noisy_h2_volume_full[
            slice_start_m:slice_end_m,
            slice_start_e:slice_end_e,
            slice_start_a:slice_end_a,
            slice_start_final_e:slice_end_final_e
        ]
        target_patch_slice = self.target_volume_full[
            slice_start_m:slice_end_m,
            slice_start_e:slice_end_e,
            slice_start_a:slice_end_a,
            slice_start_final_e:slice_end_final_e
        ]
        mask_patch_slice = self.mask_volume_full[
            slice_start_m:slice_end_m,
            slice_start_e:slice_end_e,
            slice_start_a:slice_end_a,
            slice_start_final_e:slice_end_final_e
        ]
        # Coordinate volume is (4, M, E, a, e), so slice all 4 coordinate channels
        coordinate_patch_slice = self.coordinate_volume_full[
            :, # All 4 coordinate channels
            slice_start_m:slice_end_m,
            slice_start_e:slice_end_e,
            slice_start_a:slice_end_a,
            slice_start_final_e:slice_end_final_e
        ]

        # Define padding tuple for F.pad (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
        # For 4D data (M, E, a, e), padding order is (e_left, e_right, a_left, a_right, E_left, E_right, M_left, M_right)
        padding = (pad_left_final_e, pad_right_final_e,
                   pad_left_a, pad_right_a,
                   pad_left_e, pad_right_e,
                   pad_left_m, pad_right_m)

        # Pad all patches to the desired patch_size using constant padding with 0.0
        # For noisy/target patches, add a channel dimension (unsqueeze(0)) before padding
        noisy_h1_patch = F.pad(noisy_h1_patch_slice.unsqueeze(0), padding, mode='constant', value=0.0)
        noisy_h2_patch = F.pad(noisy_h2_patch_slice.unsqueeze(0), padding, mode='constant', value=0.0)
        target_patch = F.pad(target_patch_slice.unsqueeze(0), padding, mode='constant', value=0.0)
        # Mask needs to be float for F.pad, then converted back to bool
        mask_patch = F.pad(mask_patch_slice.float().unsqueeze(0), padding, mode='constant', value=0.0).bool().squeeze(0)
        # Coordinate patch is already (4, ...) so no unsqueeze needed
        coordinate_patch = F.pad(coordinate_patch_slice, padding, mode='constant', value=0.0)

        # --- Data Augmentation: Random 4D Flips ---
        # Apply random flips along any of the M, E, a, e axes
        # Dims for flip are 1, 2, 3, 4 because of the added channel dim (0) for noisy/target
        # For mask and coordinate, dims are 0, 1, 2, 3 (mask) and 1, 2, 3, 4 (coordinate)
        
        # Consistent flip indices for all tensors (after adding channel dim for noisy/target)
        flip_dims = []
        if random.random() < 0.5: flip_dims.append(1) # Flip along M
        if random.random() < 0.5: flip_dims.append(2) # Flip along E
        if random.random() < 0.5: flip_dims.append(3) # Flip along a
        if random.random() < 0.5: flip_dims.append(4) # Flip along e

        if flip_dims:
            noisy_h1_patch = torch.flip(noisy_h1_patch, [d - 1 for d in flip_dims]) # Adjust for (C, M, E, a, e)
            noisy_h2_patch = torch.flip(noisy_h2_patch, [d - 1 for d in flip_dims])
            target_patch = torch.flip(target_patch, [d - 1 for d in flip_dims])
            mask_patch = torch.flip(mask_patch, [d - 1 for d in flip_dims]) # Mask is (M, E, a, e)
            coordinate_patch = torch.flip(coordinate_patch, [d for d in flip_dims]) # Coordinate is (4, M, E, a, e)

        # Assertions to ensure the patches have the correct final size
        expected_shape_data = (1, self.patch_M, self.patch_E, self.patch_a, self.patch_e)
        expected_shape_mask = (self.patch_M, self.patch_E, self.patch_a, self.patch_e)
        expected_shape_coord = (4, self.patch_M, self.patch_E, self.patch_a, self.patch_e)

        assert noisy_h1_patch.shape == expected_shape_data, f"Noisy H1 patch shape mismatch: Expected {expected_shape_data}, Got {noisy_h1_patch.shape}"
        assert noisy_h2_patch.shape == expected_shape_data, f"Noisy H2 patch shape mismatch: Expected {expected_shape_data}, Got {noisy_h2_patch.shape}"
        assert target_patch.shape == expected_shape_data, f"Target patch shape mismatch: Expected {expected_shape_data}, Got {target_patch.shape}"
        assert mask_patch.shape == expected_shape_mask, f"Mask patch shape mismatch: Expected {expected_shape_mask}, Got {mask_patch.shape}"
        assert coordinate_patch.shape == expected_shape_coord, f"Coord patch shape mismatch: Expected {expected_shape_coord}, Got {coordinate_patch.shape}"

        # Return the 4D patches and their original center for potential reconstruction/tracking
        return noisy_h1_patch.squeeze(0), noisy_h2_patch.squeeze(0), target_patch.squeeze(0), mask_patch, coordinate_patch, \
               (center_m, center_e, center_a, center_final_e)

# --- 3. UNet3D Model (Replaces MultiStreamCNN) ---
class ConvBlock3D(nn.Module):
    """Helper block for UNet: Conv3D -> BatchNorm3D -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UNet3D(nn.Module):
    """
    A 3D U-Net architecture for denoising 4D histogram data (processed as 5D patches).
    Input is expected to be [batch_size, channels, M, E, a, e], where channels include
    noisy_h1, noisy_h2, and positional encodings.
    """
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        # Encoder (Downsampling path)
        self.enc1 = ConvBlock3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2) # Reduces spatial dims by 2

        self.enc2 = ConvBlock3D(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock3D(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock3D(128, 256)

        # Decoder (Upsampling path)
        # Use ConvTranspose3d for learnable upsampling
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(128 + 128, 128) # +128 for skip connection

        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(64 + 64, 64) # +64 for skip connection

        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(32 + 32, 32) # +32 for skip connection

        # Output layer
        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)
        self.final_activation = nn.ReLU() # Ensure non-negative output

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        u3 = self.upconv3(b)
        # Ensure sizes match for concatenation (handle potential off-by-one from pooling/upconv)
        # If u3 and e3 don't match, crop e3. This is a common issue with U-Nets and odd dimensions.
        diff_m = e3.size(2) - u3.size(2)
        diff_e = e3.size(3) - u3.size(3)
        diff_a = e3.size(4) - u3.size(4)
        u3 = F.pad(u3, [diff_a // 2, diff_a - diff_a // 2,
                        diff_e // 2, diff_e - diff_e // 2,
                        diff_m // 2, diff_m - diff_m // 2])

        d3 = self.dec3(torch.cat((u3, e3), dim=1)) # Skip connection

        u2 = self.upconv2(d3)
        diff_m = e2.size(2) - u2.size(2)
        diff_e = e2.size(3) - u2.size(3)
        diff_a = e2.size(4) - u2.size(4)
        u2 = F.pad(u2, [diff_a // 2, diff_a - diff_a // 2,
                        diff_e // 2, diff_e - diff_e // 2,
                        diff_m // 2, diff_m - diff_m // 2])
        d2 = self.dec2(torch.cat((u2, e2), dim=1))

        u1 = self.upconv1(d2)
        diff_m = e1.size(2) - u1.size(2)
        diff_e = e1.size(3) - u1.size(3)
        diff_a = e1.size(4) - u1.size(4)
        u1 = F.pad(u1, [diff_a // 2, diff_a - diff_a // 2,
                        diff_e // 2, diff_e - diff_e // 2,
                        diff_m // 2, diff_m - diff_m // 2])
        d1 = self.dec1(torch.cat((u1, e1), dim=1))

        output = self.out_conv(d1)
        output = self.final_activation(output) # Ensure non-negative output
        return output

# --- Function to Denoise the Full 4D Volume ---
def denoise_full_volume(model, noisy_h1_full, noisy_h2_full, coords_full, patch_size, device):
    """
    Denoises the entire 4D histogram volume by processing it patch by patch
    using the trained UNet3D model. Assumes non-overlapping patches.

    Args:
        model (nn.Module): The trained UNet3D model.
        noisy_h1_full (torch.Tensor): The full 4D noisy input histogram (M, E, a, e).
        noisy_h2_full (torch.Tensor): The full 4D noisy input histogram (M, E, a, e) for the second input channel.
        coords_full (torch.Tensor): The full 4D coordinate volume (4, M, E, a, e).
        patch_size (tuple): The (patch_M, patch_E, patch_a, patch_e) dimensions.
        device (torch.device): The device (cpu or cuda) to run inference on.

    Returns:
        torch.Tensor: The full 4D denoised histogram (M, E, a, e).
    """
    model.eval() # Set model to evaluation mode
    
    # Get full volume dimensions
    M_full, E_full, A_full, final_e_full = noisy_h1_full.shape
    patch_M, patch_E, patch_A, patch_final_e = patch_size

    # Initialize an empty tensor for the reconstructed denoised volume
    denoised_full_volume = torch.zeros_like(noisy_h1_full, device=device)

    # Iterate through the volume using non-overlapping patches
    for m_start in tqdm(range(0, M_full, patch_M), desc="Denoising M-slices"):
        for e_start in range(0, E_full, patch_E):
            for a_start in range(0, A_full, patch_A):
                for final_e_start in range(0, final_e_full, patch_final_e):
                    # Define patch end coordinates
                    m_end = m_start + patch_M
                    e_end = e_start + patch_E
                    a_end = a_start + patch_A
                    final_e_end = final_e_start + patch_final_e

                    # Extract patch from noisy inputs and coordinates
                    # Add batch dimension (unsqueeze(0)) for model input
                    noisy_h1_patch = noisy_h1_full[m_start:m_end, e_start:e_end, a_start:a_end, final_e_start:final_e_end].unsqueeze(0).to(device)
                    noisy_h2_patch = noisy_h2_full[m_start:m_end, e_start:e_end, a_start:a_end, final_e_start:final_e_end].unsqueeze(0).to(device)
                    coord_patch = coords_full[:, m_start:m_end, e_start:e_end, a_start:a_end, final_e_start:final_e_end].unsqueeze(0).to(device)

                    # Ensure patches are exactly patch_size. If original volume is not perfectly divisible,
                    # the last patches might be smaller. Pad them if necessary.
                    current_patch_M, current_patch_E, current_patch_A, current_patch_final_e = noisy_h1_patch.shape[1:]

                    if (current_patch_M, current_patch_E, current_patch_A, current_patch_final_e) != patch_size:
                        # Calculate padding for each dimension (left, right)
                        pad_m = (0, patch_M - current_patch_M)
                        pad_e = (0, patch_E - current_patch_E)
                        pad_a = (0, patch_A - current_patch_A)
                        pad_final_e = (0, patch_final_e - current_patch_final_e)

                        # Padding order for F.pad is (dimN_left, dimN_right, ..., dim1_left, dim1_right)
                        # Our data is (Batch, M, E, a, e), so padding order is (e_l, e_r, a_l, a_r, E_l, E_r, M_l, M_r)
                        padding_tuple = (pad_final_e[0], pad_final_e[1],
                                         pad_a[0], pad_a[1],
                                         pad_e[0], pad_e[1],
                                         pad_m[0], pad_m[1])
                        
                        noisy_h1_patch = F.pad(noisy_h1_patch, padding_tuple, mode='constant', value=0.0)
                        noisy_h2_patch = F.pad(noisy_h2_patch, padding_tuple, mode='constant', value=0.0)
                        
                        # Coordinate patch is (Batch, 4, M, E, a, e), so padding applies to spatial dims 2,3,4,5
                        # Padding tuple for F.pad for a 6D tensor (Batch, Channels, D1, D2, D3, D4)
                        # is (D4_l, D4_r, D3_l, D3_r, D2_l, D2_r, D1_l, D1_r)
                        coord_padding_tuple = (pad_final_e[0], pad_final_e[1],
                                               pad_a[0], pad_a[1],
                                               pad_e[0], pad_e[1],
                                               pad_m[0], pad_m[1])
                        coord_patch = F.pad(coord_patch, coord_padding_tuple, mode='constant', value=0.0)

                    # --- Prepare input for UNet3D: Concatenate noisy_h1, noisy_h2, and coords ---
                    # noisy_h1_patch, noisy_h2_patch are (1, M, E, a, e)
                    # coord_patch is (1, 4, M, E, a, e)
                    # Need to unsqueeze noisy_h1/h2 to (1, 1, M, E, a, e) for concatenation
                    unet_input = torch.cat((noisy_h1_patch.unsqueeze(1), noisy_h2_patch.unsqueeze(1), coord_patch), dim=1)

                    with torch.no_grad():
                        denoised_patch = model(unet_input) # UNet3D takes single input
                    
                    # Remove batch dimension and channel dimension (output is 1 channel)
                    denoised_patch = denoised_patch.squeeze(0).squeeze(0)

                    # Crop denoised_patch back to original slice size before placing
                    denoised_patch_cropped = denoised_patch[
                        :current_patch_M, :current_patch_E, :current_patch_A, :current_patch_final_e
                    ]

                    # Place the denoised patch into the full reconstructed volume
                    denoised_full_volume[
                        m_start:m_end, e_start:e_end, a_start:a_end, final_e_start:final_e_end
                    ] = denoised_patch_cropped
    
    # --- Final Normalization of Denoised Output per (M, E) slice ---
    for m_idx in range(M_full):
        for e_idx in range(E_full):
            slice_sum = denoised_full_volume[m_idx, e_idx, :, :].sum()
            if slice_sum > 0:
                denoised_full_volume[m_idx, e_idx, :, :] /= slice_sum

    return denoised_full_volume

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    # Define synthetic data dimensions
    M_DIM_SYNTHETIC = 10
    E_DIM_SYNTHETIC = 15
    A_DIM_SYNTHETIC = 20
    e_DIM_SYNTHETIC = 25
    SYNTHETIC_NOISE_LEVEL = 0.05 # Adjusted noise_level

    # Patch size needs to be divisible by 2^num_pooling_layers for UNet.
    # With 3 pooling layers, need dimensions divisible by 8.
    # Let's adjust for better UNet compatibility or handle padding in UNet itself.
    # For now, let's make it larger to see the effect.
    PATCH_SIZE = (8, 8, 8, 8) # Adjusted for UNet pooling compatibility
    NUM_PATCHES_TO_GENERATE = 5000 # Increased patches for more training data
    BATCH_SIZE = 4 # Reduced batch size if memory is an issue with larger patches
    VALIDATION_SPLIT_RATIO = 0.2 # 20% for validation
    NUM_EPOCHS = 15 # Increased epochs for more training
    LEARNING_RATE = 0.0005 # Adjusted learning rate

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Generate Synthetic Data ---
    noisy_h1_full, noisy_h2_full_for_model_input, clean_target_full, mask_full, coords_full, \
    M_full, E_full, A_full, final_e_full = generate_synthetic_noisy_data(
        M_dim=M_DIM_SYNTHETIC,
        E_dim=E_DIM_SYNTHETIC,
        A_dim=A_DIM_SYNTHETIC,
        e_dim=e_DIM_SYNTHETIC,
        noise_level=SYNTHETIC_NOISE_LEVEL
    )

    # --- 2. Create Dataset ---
    full_dataset = DenoisingDataset(
        noisy_h1_volume=noisy_h1_full,
        noisy_h2_volume=noisy_h2_full_for_model_input,
        target_volume=clean_target_full, # This is your clean/target data
        mask_volume=mask_full,
        coordinate_volume=coords_full,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES_TO_GENERATE
    )

    # --- 3. Data Split for Validation ---
    total_samples = len(full_dataset)
    val_size = int(total_samples * VALIDATION_SPLIT_RATIO)
    train_size = total_samples - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"\nDataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train DataLoader has {len(train_dataloader)} batches.")
    print(f"Validation DataLoader has {len(val_dataloader)} batches.")

    # --- 4. Instantiate Model, Loss, and Optimizer ---
    # Input channels for UNet3D: 1 (noisy_h1) + 1 (noisy_h2) + 4 (coords) = 6 channels
    model = UNet3D(in_channels=6, out_channels=1) 
    model.to(device)

    # --- Modified: Weighted MSE Loss ---
    # Define a weight for non-zero pixels (peaks)
    PEAK_WEIGHT = 10.0 # You can adjust this value
    def weighted_mse_loss(output, target, mask, peak_weight):
        # Calculate MSE for all pixels
        mse_all = F.mse_loss(output, target, reduction='none')
        
        # Apply weight to pixels where the target (clean data) is non-zero
        # Use the mask to identify these regions
        weighted_loss = torch.where(mask, mse_all * peak_weight, mse_all)
        
        # Return the mean of the weighted loss
        return torch.mean(weighted_loss)

    criterion = weighted_mse_loss # Use the custom weighted loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5. Training Loop ---
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        running_loss = 0.0
        train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)")
        for i, (noisy_h1_patch, noisy_h2_patch, clean_target_patch, mask_patch, coord_patch, original_center_coords) in enumerate(train_loop):
            # Move data to device
            noisy_h1_patch = noisy_h1_patch.to(device)
            noisy_h2_patch = noisy_h2_patch.to(device)
            clean_target_patch = clean_target_patch.to(device)
            mask_patch = mask_patch.to(device) # Move mask to device
            coord_patch = coord_patch.to(device)

            # --- Prepare input for UNet3D: Concatenate noisy_h1, noisy_h2, and coords ---
            # noisy_h1_patch, noisy_h2_patch are (Batch, M, E, a, e)
            # coord_patch is (Batch, 4, M, E, a, e)
            # Need to unsqueeze noisy_h1/h2 to (Batch, 1, M, E, a, e) for concatenation
            unet_input = torch.cat((noisy_h1_patch.unsqueeze(1), noisy_h2_patch.unsqueeze(1), coord_patch), dim=1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(unet_input) 

            # Calculate loss using the weighted criterion
            loss = criterion(outputs.squeeze(1), clean_target_patch, mask_patch, PEAK_WEIGHT) # Squeeze output channel for loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loop.set_postfix(loss=running_loss / (i + 1))

        avg_train_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

        # --- Validation Loop ---
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculation for validation
            val_loop = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)")
            for i, (noisy_h1_patch, noisy_h2_patch, clean_target_patch, mask_patch, coord_patch, original_center_coords) in enumerate(val_loop):
                noisy_h1_patch = noisy_h1_patch.to(device)
                noisy_h2_patch = noisy_h2_patch.to(device)
                clean_target_patch = clean_target_patch.to(device)
                mask_patch = mask_patch.to(device) # Move mask to device
                coord_patch = coord_patch.to(device)

                # Prepare input for UNet3D
                unet_input = torch.cat((noisy_h1_patch.unsqueeze(1), noisy_h2_patch.unsqueeze(1), coord_patch), dim=1)

                outputs = model(unet_input)
                loss = criterion(outputs.squeeze(1), clean_target_patch, mask_patch, PEAK_WEIGHT) # Use weighted loss for validation too
                val_loss += loss.item()
                val_loop.set_postfix(val_loss=val_loss / (i + 1))

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}\n")

    print("Training complete!")
    # --- Optional: Save the trained model ---
    # torch.save(model.state_dict(), "multi_stream_denoising_model.pth")
    # print("Model saved to multi_stream_denoising_model.pth")

    # --- 6. Plotting Initial Distributions (Noisy1, Noisy2, Clean) ---
    print("\nPlotting initial data distributions for specific M, E indices...")
    noisy_h1_full_cpu = noisy_h1_full.cpu().numpy()
    noisy_h2_full_input_to_model_cpu = noisy_h2_full_for_model_input.cpu().numpy()
    clean_target_full_cpu = clean_target_full.cpu().numpy()

    # Choose specific M and E indices from the full volume for plotting initial distributions
    # Let's pick the first few indices to see the "start" of the distributions
    initial_m_indices = [0, min(1, M_full - 1)] # First two M indices
    initial_e_indices = [0, min(1, E_full - 1)] # First two E indices

    for m_idx in initial_m_indices:
        for e_idx in initial_e_indices:
            # Extract the (a,e) slices from the full volumes
            noisy1_ae_slice = noisy_h1_full_cpu[m_idx, e_idx, :, :]
            noisy2_ae_slice = noisy_h2_full_input_to_model_cpu[m_idx, e_idx, :, :]
            clean_ae_slice = clean_target_full_cpu[m_idx, e_idx, :, :]

            # Create a new figure with 1 row and 3 columns
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Initial Data Distributions (a,e) at M={m_idx}, E={e_idx}', fontsize=16)

            # Plot Noisy Input 1
            im1 = axes[0].imshow(noisy1_ae_slice, cmap='viridis', origin='lower')
            axes[0].set_title('Noisy Input 1')
            axes[0].set_xlabel('Final Energy (e)')
            axes[0].set_ylabel('Scattered Angle (a)')
            fig.colorbar(im1, ax=axes[0])

            # Plot Noisy Input 2
            im2 = axes[1].imshow(noisy2_ae_slice, cmap='viridis', origin='lower')
            axes[1].set_title('Noisy Input 2')
            axes[1].set_xlabel('Final Energy (e)')
            axes[1].set_ylabel('Scattered Angle (a)')
            fig.colorbar(im2, ax=axes[1])

            # Plot Clean Target
            im3 = axes[2].imshow(clean_ae_slice, cmap='viridis', origin='lower')
            axes[2].set_title('Clean Target')
            axes[2].set_xlabel('Final Energy (e)')
            axes[2].set_ylabel('Scattered Angle (a)')
            fig.colorbar(im3, ax=axes[2])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'initial_distribution_M{m_idx}_E{e_idx}.png') # Save the figure
            plt.close(fig) # Close the figure to free up memory


    # --- 7. Denoise and Plot Full Volume Slices (after training) ---
    print("\nDenoising full volume and plotting example slices (after training)...")
    
    # Move full data to device for inference
    noisy_h1_full_dev = noisy_h1_full.to(device)
    noisy_h2_full_for_model_input_dev = noisy_h2_full_for_model_input.to(device)
    clean_target_full_dev = clean_target_full.to(device)
    coords_full_dev = coords_full.to(device)

    # Denoise the entire volume
    denoised_full_volume = denoise_full_volume(
        model,
        noisy_h1_full_dev,
        noisy_h2_full_for_model_input_dev,
        coords_full_dev,
        PATCH_SIZE,
        device
    )

    # Move volumes back to CPU for plotting
    denoised_full_volume_cpu = denoised_full_volume.cpu().numpy()
    noisy_h1_full_cpu = noisy_h1_full.cpu().numpy()
    noisy_h2_full_input_to_model_cpu = noisy_h2_full_for_model_input.cpu().numpy()
    clean_target_full_cpu = clean_target_full.cpu().numpy()

    # Choose specific M and E indices from the full volume for plotting
    num_m_plots = min(2, M_full) # Plot up to 2 M indices
    num_e_plots = min(2, E_full) # Plot up to 2 E indices

    # Select indices to plot, ensuring they are within bounds
    plot_m_indices = np.linspace(0, M_full - 1, num=num_m_plots, dtype=int)
    plot_e_indices = np.linspace(0, E_full - 1, num=num_e_plots, dtype=int)

    for m_idx in plot_m_indices:
        for e_idx in plot_e_indices:
            # Extract the (a,e) slices from the full volumes for the current M and E
            noisy1_ae_slice = noisy_h1_full_cpu[m_idx, e_idx, :, :]
            noisy2_ae_slice = noisy_h2_full_input_to_model_cpu[m_idx, e_idx, :, :]
            clean_ae_slice = clean_target_full_cpu[m_idx, e_idx, :, :]
            denoised_ae_slice = denoised_full_volume_cpu[m_idx, e_idx, :, :]

            # Create a new figure with 1 row and 4 columns for each (M, E) combination
            fig, axes = plt.subplots(1, 4, figsize=(24, 6)) # Increased figsize for 4 plots
            fig.suptitle(f'Comparison of (a,e) Slices at M={m_idx}, E={e_idx}', fontsize=16)

            # Plot Noisy Input 1
            im1 = axes[0].imshow(noisy1_ae_slice, cmap='viridis', origin='lower')
            axes[0].set_title('Noisy Input 1')
            axes[0].set_xlabel('Final Energy (e)')
            axes[0].set_ylabel('Scattered Angle (a)')
            fig.colorbar(im1, ax=axes[0])

            # Plot Noisy Input 2
            im2 = axes[1].imshow(noisy2_ae_slice, cmap='viridis', origin='lower')
            axes[1].set_title('Noisy Input 2')
            axes[1].set_xlabel('Final Energy (e)')
            axes[1].set_ylabel('Scattered Angle (a)')
            fig.colorbar(im2, ax=axes[1])

            # Plot Clean Target
            im3 = axes[2].imshow(clean_ae_slice, cmap='viridis', origin='lower')
            axes[2].set_title('Clean Target')
            axes[2].set_xlabel('Final Energy (e)')
            axes[2].set_ylabel('Scattered Angle (a)')
            fig.colorbar(im3, ax=axes[2])

            # Plot Denoised Output
            im4 = axes[3].imshow(denoised_ae_slice, cmap='viridis', origin='lower')
            axes[3].set_title('Denoised Output')
            axes[3].set_xlabel('Final Energy (e)')
            axes[3].set_ylabel('Scattered Angle (a)')
            fig.colorbar(im4, ax=axes[3])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            plt.savefig(f'denoised_slice_M{m_idx}_E{e_idx}.png') # Save the figure
            plt.close(fig) # Close the figure to free up memory
