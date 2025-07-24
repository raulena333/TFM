import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import math
import torch.nn.functional as F # Needed for F.pad in extract_single_4d_patch

# Set device for PyTorch operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Data Generation/Preparation ---
def load_and_prepare_data(npz_path="./DenoisingDataTransSheet.npz", required_patch_size=(32, 32, 32, 32)):
    """
    Loads histogram data from an NPZ file and prepares noisy and ground truth tensors.
    Ensures data is float32.

    Args:
        npz_path (str): Path to the .npz file containing histogram data.
                        Expected to have a key "histograms" with shape (2, M, E, A, B).
                        histograms[0] is treated as noisy input, histograms[1] as ground truth.
        required_patch_size (tuple): The expected patch size (patch_M, patch_E, patch_A, patch_B).

    Returns:
        tuple: A tuple containing (noisy_histograms_np, ground_truth_histograms_np, mask_np).
                All returned arrays will be 4D (M, E, A, B) and float32.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"[!] Missing {npz_path}")

    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    histograms = data["histograms"] # Expected (2, M, E, A, B)
    print(f"Shape of loaded raw data: {histograms.shape}")

    # Validate the shape of the loaded data
    if histograms.ndim != 5 or histograms.shape[0] != 2:
        raise ValueError(
            "Expected 'histograms' data to be 5D with shape [2, M, E, A, B]. "
            f"Got shape: {histograms.shape}"
        )
    
    # Assign noisy and ground truth (which is the second noisy input for Noise2Noise)
    # Ensure float32 dtype
    noisy_input_data = histograms[0].astype(np.float32) # (M, E, A, B)
    target_data = histograms[1].astype(np.float32)      # (M, E, A, B)

    M, E, A, B = noisy_input_data.shape
    patch_M, patch_E, patch_A, patch_B = required_patch_size
    
    print(f"Loaded noisy_input_data shape: {noisy_input_data.shape}")
    print(f"Required patch size: {required_patch_size}")

    # Validate patch size against the 4 "spatial" dimensions
    if patch_M > M or patch_E > E or patch_A > A or patch_B > B:
        raise ValueError(
            f"Required patch size ({patch_M}, {patch_E}, {patch_A}, {patch_B}) "
            f"is larger than the input data's (M,E,A,B) dimensions ({M}, {E}, {A}, {B})."
        )
        
    # Create the mask based on non-zero voxels in the noisy data
    mask = (noisy_input_data > 0).astype(np.float32)
    
    print(f"Noisy volume 4D shape (M,E,A,B): {noisy_input_data.shape}")
    print(f"Target volume 4D shape (M,E,A,B): {target_data.shape}")
    print(f"Mask volume 4D shape (M,E,A,B): {mask.shape}")

    return noisy_input_data, target_data, mask

# --- 2. Patching and Augmentation Functions ---
def extract_single_4d_patch(data_4d, patch_size, nonzero_coords=None, sample_strategy_ratio=0.8):
    """
    Extracts a single 4D patch from a 4D tensor based on sampling strategy.
    Pads the patch with zeros if the data_4d dimensions are smaller than patch_size.

    Args:
        data_4d (torch.Tensor): Input tensor of shape (M, E, a, e). (Single data point, no batch dim)
        patch_size (tuple): (patch_M, patch_E, patch_a, patch_e) for patch dimensions.
        nonzero_coords (np.array, optional): N_nonzero x 4 array of (M,E,a,e) coordinates
                                            of non-zero voxels. Required for non-zero sampling.
        sample_strategy_ratio (float): Ratio (e.g., 0.8) for non-zero centered patches.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: A single patch of shape `patch_size`.
            - tuple: The (start_M, start_E, start_a, start_e) coordinates of the extracted patch.
    """
    M_dim, E_dim, a_dim, e_dim = data_4d.shape
    pM, pE, pa, pe = patch_size

    # Decide sampling method (80% non-zero, 20% random)
    if nonzero_coords is not None and len(nonzero_coords) > 0 and np.random.rand() < sample_strategy_ratio:
        # 80% case: Sample a random non-zero voxel
        idx = np.random.randint(len(nonzero_coords))
        center_coords = nonzero_coords[idx]
    else:
        # 20% case or if no non-zero voxels: Sample a completely random voxel
        center_coords = [np.random.randint(dim) for dim in data_4d.shape]

    # Calculate start coordinates for the patch to center it around the chosen voxel
    # and ensure it stays within bounds.
    start_M = max(0, center_coords[0] - pM // 2)
    start_E = max(0, center_coords[1] - pE // 2)
    start_a = max(0, center_coords[2] - pa // 2)
    start_e = max(0, center_coords[3] - pe // 2)

    # Adjust start coordinates if the patch would go out of bounds
    if start_M + pM > M_dim: start_M = M_dim - pM
    if start_E + pE > E_dim: start_E = E_dim - pE
    if start_a + pa > a_dim: start_a = a_dim - pa
    if start_e + pe > e_dim: start_e = e_dim - pe
    
    # Ensure start_coords are non-negative (important if dim < patch_size)
    start_M = max(0, start_M)
    start_E = max(0, start_E)
    start_a = max(0, start_a)
    start_e = max(0, start_e)

    # Extract the patch
    patch = data_4d[start_M : start_M + pM,
                    start_E : start_E + pE,
                    start_a : start_a + pa,
                    start_e : start_e + pe]

    # Pad if the extracted patch is smaller than the desired patch_size
    # This can happen if the original data_4d dimensions are smaller than patch_size
    if patch.shape != patch_size:
        padding_needed = []
        for i, (p_dim, d_dim) in enumerate(zip(patch_size, patch.shape)):
            pad_before = 0 # No padding at the start for extraction, only at the end
            pad_after = p_dim - d_dim
            padding_needed.extend([pad_before, pad_after])
        
        # F.pad expects padding in reverse order of dimensions (last dim first)
        padding_needed = tuple(padding_needed[::-1])
        patch = F.pad(patch, padding_needed, 'constant', 0) # Pad with zeros

    return patch, (start_M, start_E, start_a, start_e)


def apply_4d_augmentations(patch):
    """
    Applies random flips in any combination of the 4 axes and
    90-degree rotations in 2D subspaces to a 4D patch.

    Args:
        patch (torch.Tensor): Input patch of shape (patch_M, patch_E, patch_a, patch_e).

    Returns:
        torch.Tensor: Augmented patch.
    """
    # Random Flips (up to 4 axes)
    for axis in range(patch.ndim): # Iterate through all 4 dimensions
        if random.random() < 0.5: # 50% chance to flip each axis
            patch = torch.flip(patch, dims=[axis])

    # 90-degree rotations in 2D subspaces
    # There are 6 unique 2D planes in 4D: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # Corresponding to (M,E), (M,a), (M,e), (E,a), (E,e), (a,e)
    rotation_planes = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    if random.random() < 0.5: # 50% chance to apply a 2D rotation
        plane_axes = random.choice(rotation_planes)
        k = random.randint(1, 3) # Rotate by 90, 180, or 270 degrees (1, 2, or 3 times)
        patch = torch.rot90(patch, k=k, dims=plane_axes)

    return patch

# --- 3. Custom PyTorch Dataset ---
class Noisy4DPatchDataset(Dataset):
    def __init__(self, noisy_data_1_list, noisy_data_2_list, patch_size,
                 sample_strategy_ratio=0.8, num_patches_per_epoch=1000):
        """
        Args:
            noisy_data_1_list (list of np.array): List of your full noisy (M,E,a,e) volumes (noisy version 1).
            noisy_data_2_list (list of np.array): List of your full noisy (M,E,a,e) volumes (noisy version 2).
                                                 Must correspond to noisy_data_1_list.
            patch_size (tuple): (patch_M, patch_E, patch_a, patch_e).
            sample_strategy_ratio (float): Ratio for non-zero centered patches.
            num_patches_per_epoch (int): Number of patches to generate per epoch.
                                         This defines the 'length' of the dataset.
        """
        if len(noisy_data_1_list) != len(noisy_data_2_list):
            raise ValueError("noisy_data_1_list and noisy_data_2_list must have the same length.")

        self.noisy_data_1_list = [torch.from_numpy(data) for data in noisy_data_1_list]
        self.noisy_data_2_list = [torch.from_numpy(data) for data in noisy_data_2_list]
        self.patch_size = patch_size
        self.sample_strategy_ratio = sample_strategy_ratio
        self.num_patches_per_epoch = num_patches_per_epoch

        # Pre-calculate non-zero coordinates for faster sampling
        self.nonzero_coords_list = []
        for data_vol in noisy_data_1_list: # Calculate from noisy_data_1, assuming similar sparsity
            nonzero_mask = data_vol > 0 # Use > 0 for float data
            self.nonzero_coords_list.append(np.argwhere(nonzero_mask)) # np.argwhere for numpy array

    def __len__(self):
        # This defines how many patches are generated per epoch
        return self.num_patches_per_epoch

    def __getitem__(self, idx):
        # Select a random full volume from the dataset
        volume_idx = random.randint(0, len(self.noisy_data_1_list) - 1)
        
        noisy_data_1_full = self.noisy_data_1_list[volume_idx]
        noisy_data_2_full = self.noisy_data_2_list[volume_idx]
        nonzero_coords = self.nonzero_coords_list[volume_idx]

        # Extract patch for noisy_data_1 and get its coordinates
        patch_1_raw, patch_coords = extract_single_4d_patch(
            noisy_data_1_full, self.patch_size, nonzero_coords, self.sample_strategy_ratio
        )
        
        # Extract corresponding patch for noisy_data_2 using the *same* coordinates
        # We pass None for nonzero_coords and 1.0 for sample_strategy_ratio to ensure
        # it extracts strictly based on the calculated patch_coords
        patch_2_raw = noisy_data_2_full[
            patch_coords[0] : patch_coords[0] + self.patch_size[0],
            patch_coords[1] : patch_coords[1] + self.patch_size[1],
            patch_coords[2] : patch_coords[2] + self.patch_size[2],
            patch_coords[3] : patch_coords[3] + self.patch_size[3]
        ]
        # Pad patch_2_raw if it's smaller due to boundary conditions, similar to extract_single_4d_patch
        if patch_2_raw.shape != self.patch_size:
            padding_needed = []
            for i, (p_dim, d_dim) in enumerate(zip(self.patch_size, patch_2_raw.shape)):
                pad_after = p_dim - d_dim
                padding_needed.extend([0, pad_after])
            padding_needed = tuple(padding_needed[::-1])
            patch_2_raw = F.pad(patch_2_raw, padding_needed, 'constant', 0)


        # Apply augmentations *identically* to both patches
        # Use a single seed to ensure identical transformations
        seed = random.randint(0, 2**32 - 1)
        
        # Set seeds for torch and numpy for reproducible augmentations within this __getitem__ call
        torch.manual_seed(seed)
        random.seed(seed) # For random.random()
        np.random.seed(seed % (2**32 - 1)) # For np.random.rand() and np.random.randint

        augmented_patch_1 = apply_4d_augmentations(patch_1_raw)
        
        # Reset seeds for the second patch to ensure identical augmentation
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        augmented_patch_2 = apply_4d_augmentations(patch_2_raw)

        # Add a channel dimension (expected by your model's Conv3d layers)
        # Patches are (M, E, a, e), model expects (C, M, E, a, e)
        augmented_patch_1 = augmented_patch_1.unsqueeze(0) # (1, M, E, a, e)
        augmented_patch_2 = augmented_patch_2.unsqueeze(0) # (1, M, E, a, e)

        return augmented_patch_1, augmented_patch_2
    
# --- 4. Define the 4D Sequential Permutation Convolutional Model ---

class Permute3DConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,1), groups=1):
        super(Permute3DConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.InstanceNorm3d(out_channels) # InstanceNorm is often good for denoising
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class Permuting4DDenoisingNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=(16, 32, 64), initial_dims=(16, 16, 16, 16)):
        super(Permuting4DDenoisingNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_dims = initial_dims # (M, E, a, e) dimensions of the input patch
        
        # Initial convolution to increase feature depth
        # This first Conv3d will operate on (M, E, A) with B effectively as part of the batch.
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, features[0], kernel_size=(3,3,3), padding=(1,1,1)),
            nn.InstanceNorm3d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        f = features[0] # Number of feature channels after initial conv
        
        # Permuting blocks, each focusing on a different 3D subspace
        # and applying residual connections.
        
        # Block 1: Focus on (E, a, e) dimensions. M is absorbed into batch.
        # Input: (N, C, M, E, a, e) -> (N*M, C, E, a, e)
        self.block1_conv = Permute3DConvBlock(f, f, kernel_size=(3,3,3), padding=(1,1,1))
        
        # Block 2: Focus on (M, a, e) dimensions. E is absorbed into batch.
        # Input from previous block is (N, F, M, E, a, e)
        self.block2_conv = Permute3DConvBlock(f, f, kernel_size=(3,3,3), padding=(1,1,1))
        
        # Block 3: Focus on (M, E, e) dimensions. 'a' is absorbed into batch.
        self.block3_conv = Permute3DConvBlock(f, f, kernel_size=(3,3,3), padding=(1,1,1))
        
        # Block 4: Focus on (M, E, a) dimensions. 'e' is absorbed into batch.
        self.block4_conv = Permute3DConvBlock(f, f, kernel_size=(3,3,3), padding=(1,1,1))

        # Final convolution to output 1 channel
        # This will also operate on (M, E, A) with B effectively as part of the batch.
        self.final_conv = nn.Conv3d(f, out_channels, kernel_size=(1,1,1))
        
    def forward(self, x):
        # x shape: (N, 1, M, E, a, e)
        N, C, M, E, A, B = x.shape
        
        # Initial convolution: Reshape to (N*B, C, M, E, A)
        # Permute to bring 'B' (original dim 5) to position 1, then reshape.
        x_initial_reshaped = x.permute(0, 5, 1, 2, 3, 4).reshape(N * B, C, M, E, A)
        x_initial_features = self.initial_conv(x_initial_reshaped) # Output: (N*B, features[0], M, E, A)
        
        # Reshape back to (N, features[0], M, E, A, B)
        x_current = x_initial_features.reshape(N, B, self.initial_conv[0].out_channels, M, E, A)
        x_current = x_current.permute(0, 2, 3, 4, 5, 1) # (N, F, M, E, A, B)
        
        # Store initial features for residual connection
        x_residual_base = x_current 
        
        # Block 1: Conv on (E, a, e). M absorbed into batch.
        # (N, F, M, E, a, e) -> (N, M, F, E, a, e) -> (N*M, F, E, a, e)
        x1_in = x_current.permute(0, 2, 1, 3, 4, 5).reshape(N * M, x_current.shape[1], E, A, B)
        x1_out = self.block1_conv(x1_in) # (N*M, F, E_out, a_out, e_out)
        
        # Reshape back to (N, F, M, E, a, e)
        x1_out = x1_out.reshape(N, M, x1_out.shape[1], x1_out.shape[2], x1_out.shape[3], x1_out.shape[4])
        x1_out = x1_out.permute(0, 2, 1, 3, 4, 5) # (N, F, M, E, a, e)
        
        x_current = x_residual_base + x1_out # Residual connection
        
        # Block 2: Conv on (M, a, e). E absorbed into batch.
        # (N, F, M, E, a, e) -> (N, E, F, M, a, e) -> (N*E, F, M, a, e)
        x2_in = x_current.permute(0, 3, 1, 2, 4, 5).reshape(N * E, x_current.shape[1], M, A, B)
        x2_out = self.block2_conv(x2_in) # (N*E, F, M_out, a_out, e_out)
        
        # Reshape back to (N, F, M, E, a, e)
        x2_out = x2_out.reshape(N, E, x2_out.shape[1], x2_out.shape[2], x2_out.shape[3], x2_out.shape[4])
        x2_out = x2_out.permute(0, 2, 3, 1, 4, 5) # (N, F, M, E, a, e)
        
        x_current = x_current + x2_out # Residual connection
        
        # Block 3: Conv on (M, E, e). 'a' absorbed into batch.
        # (N, F, M, E, a, e) -> (N, a, F, M, E, e) -> (N*A, F, M, E, e)
        x3_in = x_current.permute(0, 4, 1, 2, 3, 5).reshape(N * A, x_current.shape[1], M, E, B)
        x3_out = self.block3_conv(x3_in) # (N*A, F, M_out, E_out, e_out)
        
        # Reshape back to (N, F, M, E, a, e)
        x3_out = x3_out.reshape(N, A, x3_out.shape[1], x3_out.shape[2], x3_out.shape[3], x3_out.shape[4])
        x3_out = x3_out.permute(0, 2, 3, 4, 1, 5) # (N, F, M, E, a, e)
        
        x_current = x_current + x3_out # Residual connection
        
        # Block 4: Conv on (M, E, a). 'e' absorbed into batch.
        # (N, F, M, E, a, e) -> (N, e, F, M, E, a) -> (N*B, F, M, E, a)
        x4_in = x_current.permute(0, 5, 1, 2, 3, 4).reshape(N * B, x_current.shape[1], M, E, A)
        x4_out = self.block4_conv(x4_in) # (N*B, F, M_out, E_out, a_out)
        
        # Reshape back to (N, F, M, E, a, e)
        x4_out = x4_out.reshape(N, B, x4_out.shape[1], x4_out.shape[2], x4_out.shape[3], x4_out.shape[4])
        x4_out = x4_out.permute(0, 2, 3, 4, 5, 1) # (N, F, M, E, a, e)
        
        x_current = x_current + x4_out # Residual connection

        # Final convolution layer: Reshape to (N*B, F, M, E, A)
        final_conv_in = x_current.permute(0, 5, 1, 2, 3, 4).reshape(N * B, x_current.shape[1], M, E, A)
        output = self.final_conv(final_conv_in) # (N*B, out_channels, M, E, A)
        
        # Reshape back to (N, out_channels, M, E, A, B)
        output = output.reshape(N, B, self.out_channels, M, E, A)
        output = output.permute(0, 2, 3, 4, 5, 1) # (N, out_channels, M, E, A, B)
            
        return output

# --- 5. Inference Function for Full Histogram Denoising ---

def denoise_full_4d_histogram(model, noisy_full_volume, patch_size, overlap_ratio=0.5):
    """
    Denoises an entire 4D histogram using a patch-based deep learning model.

    Args:
        model (nn.Module): The trained PyTorch denoising model.
                           Expects input of shape (N, C, D, H, W) for Conv3d layers.
        noisy_full_volume (torch.Tensor): The 4D input histogram to denoise (M, E, A, B).
                                         Expected to be on CPU.
        patch_size (tuple): (patch_M, patch_E, patch_A, patch_B) - size of patches used by the model.
        overlap_ratio (float): The ratio of overlap between adjacent patches (0 to 1).
                               E.g., 0.5 means 50% overlap.

    Returns:
        torch.Tensor: The denoised 4D histogram of the same shape as noisy_full_volume.
    """
    model.eval() # Set model to evaluation mode
    
    # Ensure input volume is on CPU and float32
    noisy_full_volume = noisy_full_volume.to('cpu').float()
    
    M, E, A, B = noisy_full_volume.shape
    pM, pE, pA, pB = patch_size

    # Calculate strides based on overlap_ratio
    # stride = patch_size * (1 - overlap_ratio)
    # Ensure strides are at least 1
    sM = max(1, int(pM * (1 - overlap_ratio)))
    sE = max(1, int(pE * (1 - overlap_ratio)))
    sA = max(1, int(pA * (1 - overlap_ratio)))
    sB = max(1, int(pB * (1 - overlap_ratio)))

    # Initialize output volume and counter for overlap-add
    denoised_volume = torch.zeros_like(noisy_full_volume)
    overlap_counter = torch.zeros_like(noisy_full_volume)

    # Calculate number of patches along each dimension
    # Add 1 to ensure coverage even if not perfectly divisible
    num_patches_M = (M - pM) // sM + 1 if M > pM else 1
    num_patches_E = (E - pE) // sE + 1 if E > pE else 1
    num_patches_A = (A - pA) // sA + 1 if A > pA else 1
    num_patches_B = (B - pB) // sB + 1 if B > pB else 1
    
    # If the dimension is smaller than the patch size, we'll only take one patch covering the whole dim
    if M <= pM: num_patches_M = 1
    if E <= pE: num_patches_E = 1
    if A <= pA: num_patches_A = 1
    if B <= pB: num_patches_B = 1


    print(f"\nDenoising full volume of shape {noisy_full_volume.shape} with patches of size {patch_size} and overlap ratio {overlap_ratio}.")
    print(f"Calculated strides: ({sM}, {sE}, {sA}, {sB})")
    print(f"Number of patches: ({num_patches_M}, {num_patches_E}, {num_patches_A}, {num_patches_B})")

    with torch.no_grad(): # Disable gradient calculation for inference
        for iM in tqdm(range(num_patches_M), desc="Processing M-dim"):
            for iE in range(num_patches_E):
                for iA in range(num_patches_A):
                    for iB in range(num_patches_B):
                        # Calculate start and end coordinates for the patch
                        start_M = iM * sM
                        end_M = start_M + pM
                        start_E = iE * sE
                        end_E = start_E + pE
                        start_A = iA * sA
                        end_A = start_A + pA
                        start_B = iB * sB
                        end_B = start_B + pB

                        # Adjust start coordinates for boundary cases to ensure patch fits
                        # This means the last patch might start earlier to fit the end of the volume
                        if end_M > M: start_M = M - pM
                        if end_E > E: start_E = E - pE
                        if end_A > A: start_A = A - pA
                        if end_B > B: start_B = B - pB
                        
                        # Ensure non-negative start (should be handled by max(0, ..) if initial check)
                        start_M = max(0, start_M)
                        start_E = max(0, start_E)
                        start_A = max(0, start_A)
                        start_B = max(0, start_B)

                        # Extract patch from the noisy volume
                        patch = noisy_full_volume[start_M:end_M, start_E:end_E, start_A:end_A, start_B:end_B]
                        
                        # Pad the patch if its actual size is smaller than patch_size
                        # This can happen at boundaries. Pad only at the end.
                        current_patch_shape = patch.shape
                        if current_patch_shape != patch_size:
                            padding_needed = []
                            for i, (p_dim, c_dim) in enumerate(zip(patch_size, current_patch_shape)):
                                pad_after = p_dim - c_dim
                                padding_needed.extend([0, pad_after]) # Pad only at the end
                            padding_needed = tuple(padding_needed[::-1]) # F.pad expects reverse order
                            patch = F.pad(patch, padding_needed, 'constant', 0)
                        
                        # Add channel dimension and move to device
                        patch_input_model = patch.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, pM, pE, pA, pB)

                        # Denoise the patch
                        denoised_patch_output = model(patch_input_model) # (1, 1, pM, pE, pA, pB)
                        
                        # Remove batch and channel dimensions and move back to CPU
                        denoised_patch_output = denoised_patch_output.squeeze(0).squeeze(0).cpu() # (pM, pE, pA, pB)

                        # Add the denoised patch to the output volume and update counter
                        # Use the actual *extracted* size for placing the patch back, not the padded size
                        # This ensures only the valid part of the denoised patch is used.
                        actual_end_M = start_M + current_patch_shape[0]
                        actual_end_E = start_E + current_patch_shape[1]
                        actual_end_A = start_A + current_patch_shape[2]
                        actual_end_B = start_B + current_patch_shape[3]

                        denoised_volume[start_M:actual_end_M, start_E:actual_end_E, start_A:actual_end_A, start_B:actual_end_B] += \
                            denoised_patch_output[:current_patch_shape[0], :current_patch_shape[1], :current_patch_shape[2], :current_patch_shape[3]]
                            
                        overlap_counter[start_M:actual_end_M, start_E:actual_end_E, start_A:actual_end_A, start_B:actual_end_B] += 1

    # Divide by overlap_counter to get the average for overlapping regions
    # Avoid division by zero for areas not covered by any patch (shouldn't happen with overlap)
    # Set any zero counts to 1 to avoid NaN results, though ideally all regions are covered.
    overlap_counter[overlap_counter == 0] = 1 
    final_denoised_volume = denoised_volume / overlap_counter
    
    return final_denoised_volume

# --- Main Execution Block ---
if __name__ == '__main__':
    # Define global patch size for consistency
    GLOBAL_PATCH_SIZE = (8, 8, 8, 8) # (patch_M, patch_E, patch_a, patch_e)
    energyRange = (-0.6, 0)
    angleRange = (0, 70)
    
    extent = [angleRange[0], angleRange[1], energyRange[0], energyRange[1]]

    # Create a dummy NPZ file for testing if it doesn't exist
    dummy_npz_path = "./DenoisingDataTransSheet.npz"
    if not os.path.exists(dummy_npz_path):
        # This block creates a dummy .npz file if it doesn't exist.
        # It generates two 4D arrays (50, 50, 100, 100) as noisy and target.
        # This allows the script to run without requiring a pre-existing file for testing.
        print(f"Creating dummy NPZ file: {dummy_npz_path}")
        dummy_noisy_data = np.random.rand(50, 50, 100, 100).astype(np.float32)
        dummy_target_data = dummy_noisy_data + np.random.rand(50, 50, 100, 100).astype(np.float32) * 0.1 # Add some noise
        # Ensure some zeros for mask creation
        dummy_noisy_data[dummy_noisy_data < 0.1] = 0 
        dummy_target_data[dummy_target_data < 0.1] = 0

        histograms_dummy = np.stack([dummy_noisy_data, dummy_target_data], axis=0)
        np.savez(dummy_npz_path, histograms=histograms_dummy)
        print(f"Dummy NPZ file created at {dummy_npz_path}")

    try:
        # Load and prepare data using your provided function
        # This will return numpy arrays
        noisy_input_np, target_np, mask_np = load_and_prepare_data(
            npz_path=dummy_npz_path,
            required_patch_size=GLOBAL_PATCH_SIZE
        )

        # Wrap the single loaded volume in a list for the Dataset
        noisy_data_1_list = [noisy_input_np]
        noisy_data_2_list = [target_np] # This is the second noisy version for Noise2Noise

        # Create the dataset
        dataset = Noisy4DPatchDataset(
            noisy_data_1_list,
            noisy_data_2_list,
            patch_size=GLOBAL_PATCH_SIZE,
            sample_strategy_ratio=0.8,
            num_patches_per_epoch=10000 # <-- INCREASED for full training
        )

        # Create the DataLoader
        batch_size = 4 # <-- Adjusted for potentially better gradient stability
        num_workers_to_use = min(os.cpu_count(), 4) # Use up to 4 workers, or less if CPU count is low
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_to_use, pin_memory=True) 

        print(f"\nDataset length (patches per epoch): {len(dataset)}")
        print(f"DataLoader batch size: {dataloader.batch_size}")
        print(f"DataLoader num_workers: {dataloader.num_workers}")

        # Test fetching a batch
        print("\nTesting one batch from DataLoader:")
        first_batch_x, first_batch_y = next(iter(dataloader)) # Get the first batch
        
        print(f"First batch - Input (X) shape: {first_batch_x.shape}")
        print(f"First batch - Target (Y) shape: {first_batch_y.shape}")
        
        if first_batch_x.shape[2:] != GLOBAL_PATCH_SIZE:
            print(f"Warning: Patch size mismatch. Expected {GLOBAL_PATCH_SIZE}, got {first_batch_x.shape[2:]}")
                
        print("\nData preparation and patching setup complete.")

    except FileNotFoundError as e:
        print(e)
        print("Please ensure 'DenoisingDataTransSheet.npz' exists or adjust the dummy data creation.")
        exit() # Exit if data loading fails
    except ValueError as e:
        print(e)
        exit() # Exit if data validation fails


    # Model instantiation
    model = Permuting4DDenoisingNet(
        in_channels=1, # Our DataLoader outputs (1, M, E, a, e)
        out_channels=1,
        features=(32, 64, 128), # <-- INCREASED model capacity
        initial_dims=GLOBAL_PATCH_SIZE
    ).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss() # Mean Squared Error for denoising
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Standard learning rate

    # Training parameters
    num_epochs = 50 # <-- INCREASED for full training
    log_interval = 100 # Log loss every N batches

    print("\nStarting full training...")
    model.train() # Set model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        # Use tqdm for a progress bar during training
        for batch_idx, (noisy_input, target_output) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            noisy_input = noisy_input.to(device)
            target_output = target_output.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(noisy_input)
            
            # Ensure output shape matches target shape before calculating loss
            # This check is important, as any dimension mismatch will cause errors.
            # If `outputs.shape[2:]` is still not `GLOBAL_PATCH_SIZE` after ensuring
            # kernel/padding are (3,3,3) and (1,1,1), then there's a logic error
            # in permutation/reshaping. The interpolation is a fallback for testing.
            if outputs.shape != target_output.shape:
                 print(f"Warning: Shape mismatch! Outputs: {outputs.shape}, Target: {target_output.shape}. Attempting interpolation.")
                 outputs = F.interpolate(outputs, size=target_output.shape[2:], mode='trilinear', align_corners=False)

            loss = criterion(outputs, target_output)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % log_interval == 0:
                # Log average loss over the last 'log_interval' batches
                print(f"  Batch {batch_idx+1}/{len(dataloader)}, Avg Loss: {running_loss / log_interval:.6f}")
                running_loss = 0.0
                
        avg_epoch_loss = running_loss / (len(dataloader) % log_interval if len(dataloader) % log_interval != 0 else log_interval) if running_loss > 0 else 0
        print(f"Epoch {epoch+1} finished. Total Samples Processed: {len(dataloader) * dataloader.batch_size}. Average Epoch Loss: {avg_epoch_loss:.6f}")

    print("\nTraining complete!")

    # --- 6. (Optional) Inference / Visualization (after training) ---
    # You can add code here to test the model on a sample patch and visualize results.
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        sample_noisy_input, sample_target_output = next(iter(dataloader)) # Get a fresh batch
        sample_noisy_input = sample_noisy_input.to(device)
        
        denoised_output_patch = model(sample_noisy_input)
        
        # Move back to CPU for visualization
        sample_noisy_input_cpu = sample_noisy_input[0, 0].cpu().numpy() # Take first sample, first channel
        sample_target_output_cpu = sample_target_output[0, 0].cpu().numpy()
        denoised_output_patch_cpu = denoised_output_patch[0, 0].cpu().numpy()

        print(f"\nSample denoised patch output shape: {denoised_output_patch_cpu.shape}")

        # Visualization of a single patch slice
        slice_M_idx_patch = sample_noisy_input_cpu.shape[0] // 2
        slice_E_idx_patch = sample_noisy_input_cpu.shape[1] // 2

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(sample_noisy_input_cpu[slice_M_idx_patch, slice_E_idx_patch, :, :], cmap='viridis', extent=extent, aspect="auto", origin="lower")
        plt.title(f'Noisy Input Patch (M={slice_M_idx_patch}, E={slice_E_idx_patch} slice)')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow(sample_target_output_cpu[slice_M_idx_patch, slice_E_idx_patch, :, :], cmap='viridis', extent=extent, aspect="auto", origin="lower")
        plt.title(f'Target Patch (Noisy B) (M={slice_M_idx_patch}, E={slice_E_idx_patch} slice)')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.imshow(denoised_output_patch_cpu[slice_M_idx_patch, slice_E_idx_patch, :, :], cmap='viridis', extent=extent, aspect="auto", origin="lower")
        plt.title(f'Denoised Output Patch (M={slice_M_idx_patch}, E={slice_E_idx_patch} slice)')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('./DenoisingExample.pdf')
        plt.close()

    # --- 7. Demonstration of Full Histogram Denoising ---
    
    print("\n--- Demonstrating Full Histogram Denoising ---")
    
    # Use the 'noisy_input_np' loaded earlier as the full volume for inference
    # Ensure it's a PyTorch tensor on CPU
    full_noisy_data_tensor = torch.from_numpy(noisy_input_np).float().cpu()
    
    print(f"Full noisy histogram shape for inference: {full_noisy_data_tensor.shape}")
    
    # Denoise the full histogram
    denoised_full_histogram = denoise_full_4d_histogram(
        model=model,
        noisy_full_volume=full_noisy_data_tensor,
        patch_size=GLOBAL_PATCH_SIZE, # Use the same patch size as during training
        overlap_ratio=0.5 # 50% overlap for reconstruction
    )

    print(f"Denoised full histogram shape: {denoised_full_histogram.shape}")

    # --- 8. Visualization of Full Histogram Slices (after denoising) ---
    # Take a representative slice (e.g., M=mid, E=mid) and visualize (a, e) plane
    
    full_M_idx = full_noisy_data_tensor.shape[0] // 2
    full_E_idx = full_noisy_data_tensor.shape[1] // 2
    
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(full_noisy_data_tensor[full_M_idx, full_E_idx, :, :].numpy(), cmap='viridis', extent=extent, aspect="auto", origin="lower")
    plt.title(f'Full Noisy Input (M={full_M_idx}, E={full_E_idx} slice)')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    # The 'target_np' corresponds to the second noisy input, not a clean ground truth
    # If you have a truly clean version, use that here. For Noise2Noise, this is fine.
    plt.imshow(target_np[full_M_idx, full_E_idx, :, :], cmap='viridis', extent=extent, aspect="auto", origin="lower")
    plt.title(f'Full Target (Noisy B) (M={full_M_idx}, E={full_E_idx} slice)')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(denoised_full_histogram[full_M_idx, full_E_idx, :, :].numpy(), cmap='viridis', extent=extent, aspect="auto", origin="lower")
    plt.title(f'Full Denoised Output (M={full_M_idx}, E={full_E_idx} slice)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('./FullDenoisingExample.pdf')
    plt.close()