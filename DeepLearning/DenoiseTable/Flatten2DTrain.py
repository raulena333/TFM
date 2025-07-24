import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import math
from tqdm import tqdm # Import tqdm for the progress bar
import matplotlib.pyplot as plt # Import matplotlib for plotting

# Ensure reproducibility for consistent results
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Global variables to store original 'a' and 'e' dimensions for reshaping later
ORIGINAL_A = None
ORIGINAL_E = None

# --- 1. Data Generation/Preparation ---
def load_and_prepare_data(npz_path="./DenoisingDataTransSheet.npz", required_patch_size=(32, 32, 32)):
    """
    Loads 5D data from an NPZ file, flattens it to 3D volumes,
    and generates a mask based on non-zero voxels in the noisy data.

    Args:
        npz_path (str): Path to the .npz file containing the 'histograms' data.
        required_patch_size (tuple): The minimum required patch size (D, H, W)
                                     for the U-Net architecture.

    Returns:
        tuple: (noisy_volume_3d, target_volume_3d, mask_volume_3d, original_a, original_e)
               as PyTorch tensors and integers.
    """
    global ORIGINAL_A, ORIGINAL_E # Declare intent to modify global variables

    print(f"Loading data from: {npz_path}")
    try:
        loaded_data = np.load(npz_path)
        # Assuming the noisy 5D data is stored under the key 'histograms'
        histograms_np = loaded_data['histograms']
        print(f"Loaded 'histograms' data with shape: {histograms_np.shape}")

        # Validate the shape of the loaded data
        if histograms_np.ndim != 5 or histograms_np.shape[0] != 2:
            raise ValueError(
                "Expected 'histograms' data to be 5D with shape [2, M, E, a, e]. "
                f"Got shape: {histograms_np.shape}"
            )

        # Explicitly cast the loaded NumPy array to float32
        histograms_np = histograms_np.astype(np.float32)

        # Extract noisy and target 4D datasets from the 5D data
        # histograms_np[0] is noisydata1, histograms_np[1] is noisydata2 (target)
        noisydata1_4d = histograms_np[0] # Shape: [M, E, a, e]
        noisydata2_4d = histograms_np[1] # Shape: [M, E, a, e] (This will be our target for denoising)

        # Get original dimensions for clarity and store a, e globally
        M, E, a, e = noisydata1_4d.shape
        ORIGINAL_A = a
        ORIGINAL_E = e

        # Flatten the last two dimensions (a, e) to (a*e)
        # Resulting shape: [M, E, a*e]
        noisydata1_3d = noisydata1_4d.reshape(M, E, a * e)
        noisydata2_3d = noisydata2_4d.reshape(M, E, a * e)

        # Check if volume dimensions are sufficient for the patch size
        vol_D, vol_H, vol_W = noisydata1_3d.shape
        patch_D, patch_H, patch_W = required_patch_size
        if vol_D < patch_D or vol_H < patch_H or vol_W < patch_W:
            print(f"Warning: Loaded volume dimensions ({vol_D}, {vol_H}, {vol_W}) "
                  f"are smaller than the required patch size ({patch_D}, {patch_H}, {patch_W}).")
            print("This might lead to patches being entirely padding or unexpected behavior.")
            # Consider resizing or padding the entire volume if this is a common issue for your data.

        # Create the mask based on non-zero voxels in the noisy data (noisydata1_3d).
        # This mask will be used to guide the 80% patch sampling strategy.
        mask = (noisydata1_3d != 0) # True where noisydata1_3d is not zero, False otherwise

        print(f"Noisy Volume (3D) Shape: {noisydata1_3d.shape}")
        print(f"Target Volume (3D) Shape: {noisydata2_3d.shape}")
        print(f"Mask Volume (3D) Shape: {mask.shape} (derived from non-zero voxels)")

        # Convert NumPy arrays to PyTorch tensors (they are already float32 now)
        return torch.from_numpy(noisydata1_3d), torch.from_numpy(noisydata2_3d), torch.from_numpy(mask), a, e

    except FileNotFoundError:
        print(f"Error: The file '{npz_path}' was not found.")
        print("Please ensure the .npz file is in the correct directory.")
        # Fallback to dummy data if file not found, or raise an error
        print("Generating dummy data for demonstration purposes as a fallback.")
        M, E, a, e = 64, 64, 10, 10 # Default dummy dimensions
        # If the original dimensions were smaller than required_patch_size,
        # ensure dummy data is at least required_patch_size for model compatibility.
        M = max(M, required_patch_size[0])
        E = max(E, required_patch_size[1])
        ae_flattened_dim = a * e
        ae_flattened_dim = max(ae_flattened_dim, required_patch_size[2])
        
        data_5d = np.random.rand(2, M, E, a, e).astype(np.float32) # Ensure dummy data is float32
        noisydata1_4d = data_5d[0]
        noisydata2_4d = data_5d[1]
        noisydata1_3d = noisydata1_4d.reshape(M, E, ae_flattened_dim)
        noisydata2_3d = noisydata2_4d.reshape(M, E, ae_flattened_dim)
        # Generate mask from dummy data
        mask = (noisydata1_3d != 0)
        
        # Store original a, e for dummy data as well
        ORIGINAL_A = a
        ORIGINAL_E = e
        return torch.from_numpy(noisydata1_3d), torch.from_numpy(noisydata2_3d), torch.from_numpy(mask), a, e
    except Exception as e:
        print(f"An error occurred while loading or processing data: {e}")
        print("Generating dummy data for demonstration purposes as a fallback.")
        M, E, a, e = 64, 64, 10, 10 # Default dummy dimensions
        # If the original dimensions were smaller than required_patch_size,
        # ensure dummy data is at least required_patch_size for model compatibility.
        M = max(M, required_patch_size[0])
        E = max(E, required_patch_size[1])
        ae_flattened_dim = a * e
        ae_flattened_dim = max(ae_flattened_dim, required_patch_size[2])

        data_5d = np.random.rand(2, M, E, a, e).astype(np.float32) # Ensure dummy data is float32
        noisydata1_4d = data_5d[0]
        noisydata2_4d = data_5d[1]
        noisydata1_3d = noisydata1_4d.reshape(M, E, ae_flattened_dim)
        noisydata2_3d = noisydata2_4d.reshape(M, E, ae_flattened_dim)
        # Generate mask from dummy data
        mask = (noisydata1_3d != 0)

        # Store original a, e for dummy data as well
        ORIGINAL_A = a
        ORIGINAL_E = e
        return torch.from_numpy(noisydata1_3d), torch.from_numpy(noisydata2_3d), torch.from_numpy(mask), a, e


# Define the patch size that is compatible with the U-Net depth (e.g., 32x32x32 for 4 pooling layers)
# This is a critical change to resolve the "output size too small" error.
# If your actual data volumes are smaller than this, you might need to adjust the U-Net depth
# (i.e., reduce the number of feature levels in the UNet3D constructor).
PATCH_SIZE = (32, 32, 32)

# Load your actual data and capture original 'a' and 'e' dimensions
noisy_volume, target_volume, mask_volume, original_a_dim, original_e_dim = load_and_prepare_data(npz_path="./DenoisingDataTransSheet.npz", required_patch_size=PATCH_SIZE)

# --- 2. Patch Extraction and Augmentations ---
class VolumePatchDataset(Dataset):
    """
    A PyTorch Dataset for extracting 3D patches from volumes.
    It supports specified patch size, number of patches, and a sampling strategy
    (80% centered on non-zero mask voxels, 20% random).
    It also applies random flips and 90-degree rotations as augmentations.
    """
    def __init__(self, noisy_volume, target_volume, mask_volume, patch_size=PATCH_SIZE, num_patches=10000, random_sampling_ratio=0.2):
        self.noisy_volume = noisy_volume # The input volume (e.g., noisy data)
        self.target_volume = target_volume # The ground truth volume (e.g., clean data)
        self.mask_volume = mask_volume # Mask indicating regions of interest for sampling
        self.patch_size = patch_size # Desired dimensions of the extracted patches (D, H, W)
        self.num_patches = num_patches # Total number of patches to generate for the dataset
        self.random_sampling_ratio = random_sampling_ratio # Proportion of patches to be sampled randomly
        self.volume_shape = noisy_volume.shape # Shape of the original 3D volume (D, H, W)

        # Pre-compute coordinates of non-zero voxels in the mask for efficient sampling
        self.non_zero_coords = torch.nonzero(mask_volume, as_tuple=False).tolist()

        if not self.non_zero_coords:
            print("Warning: Mask has no non-zero voxels. All patches will be randomly sampled.")

    def __len__(self):
        """Returns the total number of patches in the dataset."""
        return self.num_patches

    def __getitem__(self, idx):
        """
        Retrieves a single patch pair (noisy_patch, target_patch) with augmentations.
        The patch is extracted based on the sampling strategy.
        """
        patch_d, patch_h, patch_w = self.patch_size
        D, H, W = self.volume_shape

        # Determine the center of the patch based on the sampling strategy
        if random.random() < self.random_sampling_ratio or not self.non_zero_coords:
            # 20% of patches: fully random sampling (including pure zeros)
            # Calculate the valid range for the center of the patch
            # A patch of size P centered at C has its start at C - P//2 and end at C + P - P//2.
            # So, C - P//2 >= 0  => C >= P//2
            # And C + P - P//2 <= D => C <= D - (P - P//2)
            min_center_d = patch_d // 2
            max_center_d = D - (patch_d - patch_d // 2)
            min_center_h = patch_h // 2
            max_center_h = H - (patch_h - patch_h // 2)
            min_center_w = patch_w // 2
            max_center_w = W - (patch_w - patch_w // 2)

            # Ensure the range for randint is valid (lower_bound <= upper_bound)
            # If the volume dimension is smaller than the patch dimension,
            # min_center will be greater than max_center. In this case,
            # we clamp max_center to min_center to make the range valid for randint.
            # The padding logic later will handle the smaller extracted patch.
            center_d = random.randint(min_center_d, max(min_center_d, max_center_d))
            center_h = random.randint(min_center_h, max(min_center_h, max_center_h))
            center_w = random.randint(min_center_w, max(min_center_w, max_center_w))
        else:
            # 80% of patches: centered on a non-zero voxel (via the mask)
            # Randomly select a coordinate from the pre-computed non-zero mask coordinates
            coord = random.choice(self.non_zero_coords)
            center_d, center_h, center_w = coord[0], coord[1], coord[2]

        # Calculate the desired start and end indices for a patch of patch_size
        # centered at the chosen coordinates.
        desired_d_start = center_d - patch_d // 2
        desired_d_end = desired_d_start + patch_d
        desired_h_start = center_h - patch_h // 2
        desired_h_end = desired_h_start + patch_h
        desired_w_start = center_w - patch_w // 2
        desired_w_end = desired_w_start + patch_w

        # Calculate the actual slice indices by clamping to volume bounds
        actual_d_start = max(0, desired_d_start)
        actual_d_end = min(D, desired_d_end)
        actual_h_start = max(0, desired_h_start)
        actual_h_end = min(H, desired_h_end)
        actual_w_start = max(0, desired_w_start)
        actual_w_end = min(W, desired_w_end)

        # Extract the raw patches from the volumes
        noisy_patch_raw = self.noisy_volume[actual_d_start:actual_d_end,
                                            actual_h_start:actual_h_end,
                                            actual_w_start:actual_w_end]
        target_patch_raw = self.target_volume[actual_d_start:actual_d_end,
                                              actual_h_start:actual_h_end,
                                              actual_w_start:actual_w_end]

        # Calculate padding needed for each side to make the patch exactly `patch_size`
        # F.pad expects padding in (left, right, top, bottom, front, back) order for 3D
        pad_w_before = actual_w_start - desired_w_start
        pad_w_after = desired_w_end - actual_w_end
        pad_h_before = actual_h_start - desired_h_start
        pad_h_after = desired_h_end - actual_h_end
        pad_d_before = actual_d_start - desired_d_start
        pad_d_after = desired_d_end - actual_d_end

        # Apply padding to make the patch exactly `patch_size`
        noisy_patch = F.pad(noisy_patch_raw, (pad_w_before, pad_w_after,
                                              pad_h_before, pad_h_after,
                                              pad_d_before, pad_d_after))
        target_patch = F.pad(target_patch_raw, (pad_w_before, pad_w_after,
                                                pad_h_before, pad_h_after,
                                                pad_d_before, pad_d_after))

        # --- Augmentations ---
        # 1. Random flips in any combination of the 3 dimensions
        if random.random() < 0.5: # Flip along depth (0)
            noisy_patch = torch.flip(noisy_patch, dims=[0])
            target_patch = torch.flip(target_patch, dims=[0])
        if random.random() < 0.5: # Flip along height (1)
            noisy_patch = torch.flip(noisy_patch, dims=[1])
            target_patch = torch.flip(target_patch, dims=[1])
        if random.random() < 0.5: # Flip along width (2)
            noisy_patch = torch.flip(noisy_patch, dims=[2])
            target_patch = torch.flip(target_patch, dims=[2])

        # 2. 90° rotations in 2D subspaces
        if random.random() < 0.5:
            # Randomly pick two dimensions to rotate (e.g., XY, XZ, YZ planes)
            dims_to_rotate = random.sample([0, 1, 2], 2)
            k = random.randint(1, 3) # Rotate 90, 180, or 270 degrees (k=1, 2, or 3)
            noisy_patch = torch.rot90(noisy_patch, k=k, dims=dims_to_rotate)
            target_patch = torch.rot90(target_patch, k=k, dims=dims_to_rotate)

        # Add a channel dimension (PyTorch Conv3D expects input shape: [N, C, D, H, W])
        noisy_patch = noisy_patch.unsqueeze(0) # Shape: [1, D, H, W]
        target_patch = target_patch.unsqueeze(0) # Shape: [1, D, H, W]

        return noisy_patch, target_patch

# Create dataset and dataloader for training
patch_dataset = VolumePatchDataset(noisy_volume, target_volume, mask_volume, num_patches=50000, patch_size=PATCH_SIZE)
# Use num_workers=0 for simpler debugging; for production, increase based on CPU cores.
patch_dataloader = DataLoader(patch_dataset, batch_size=4, shuffle=True, num_workers=0)

print(f"Dataset created with {len(patch_dataset)} patches.")
# Test a batch from the dataloader to verify shapes
for i, (noisy_p, target_p) in enumerate(patch_dataloader):
    print(f"Sample Batch {i+1} - Noisy Patch Shape: {noisy_p.shape}, Target Patch Shape: {target_p.shape}")
    if i == 0: # Only print for the first batch
        break

# --- 3. U-Net Model Definition ---
class ConvBlock3D(nn.Module):
    """
    A standard 3D Convolutional Block used in the U-Net.
    Consists of two Conv3D layers, each followed by BatchNorm3d and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UpConvBlock3D(nn.Module):
    """
    A 3D Up-convolutional (Transpose Convolution) block for the decoding path.
    Used to increase spatial dimensions.
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(UpConvBlock3D, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    def forward(self, x):
        return self.upconv(x)

class UNet3D(nn.Module):
    """
    A 3D U-Net architecture for volumetric image segmentation/denoising.
    It consists of an encoder path, a bottleneck, and a decoder path with skip connections.
    """
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features # Defines the number of feature maps at each level

        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2) # Downsampling layer
        for i, feature in enumerate(features):
            if i == 0:
                self.encoder_blocks.append(ConvBlock3D(in_channels, feature))
            else:
                self.encoder_blocks.append(ConvBlock3D(features[i-1], feature))

        # Bottleneck layer
        self.bottleneck = ConvBlock3D(features[-1], features[-1] * 2)

        # Decoder path
        self.decoder_blocks = nn.ModuleList()
        self.upconv_blocks = nn.ModuleList()
        # Iterate in reverse order of features for the decoder
        for i in reversed(range(len(features))):
            if i == len(features) - 1: # First up-convolution from bottleneck
                self.upconv_blocks.append(UpConvBlock3D(features[-1] * 2, features[-1]))
                # Concatenation of bottleneck output and skip connection (features[-1]*2)
                self.decoder_blocks.append(ConvBlock3D(features[-1] * 2, features[-1]))
            else: # Subsequent up-convolutions
                self.upconv_blocks.append(UpConvBlock3D(features[i+1], features[i]))
                # Concatenation of previous decoder output and skip connection (features[i+1])
                self.decoder_blocks.append(ConvBlock3D(features[i+1], features[i]))

        # Final 1x1x1 convolution to get the desired output channels
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = [] # To store outputs from encoder for skip connections
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x) # Store the output before pooling
            x = self.pool(x) # Apply pooling for downsampling

        x = self.bottleneck(x) # Pass through the bottleneck

        skip_connections = skip_connections[::-1] # Reverse the list for decoding path

        for i in range(len(self.upconv_blocks)):
            x = self.upconv_blocks[i](x) # Upsample the feature map

            # Adjust size if necessary due to odd input dimensions or pooling effects.
            # This ensures that the upsampled feature map matches the corresponding
            # skip connection's spatial dimensions for concatenation.
            target_size = skip_connections[i].shape[2:]
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)

            # Concatenate the upsampled feature map with the skip connection
            x = torch.cat((skip_connections[i], x), dim=1) # Concatenate along the channel dimension
            x = self.decoder_blocks[i](x) # Pass through the decoder convolutional block

        return self.final_conv(x) # Final convolution to produce the output

# Instantiate the U-Net model with features chosen to keep parameters in range
model = UNet3D(in_channels=1, out_channels=1, features=[16, 32, 64, 128]) # [32, 64, 128, 256]
# Calculate and print the total number of trainable parameters in the model
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters in UNet3D: {num_params}")
# Expected parameters with features=[16, 32, 64, 128] should be around 1.8M, fitting the 3-5M range.

# Check if CUDA (GPU) is available and move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model moved to device: {device}")

# --- 4. Training Loop ---
def train_model(model, dataloader, epochs=5, learning_rate=1e-4):
    """
    Trains the U-Net model using the provided dataloader.
    """
    criterion = nn.MSELoss() # Mean Squared Error Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer

    model.train() # Set the model to training mode
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (noisy_patches, target_patches) in enumerate(dataloader):
            # Move data to the specified device (CPU or GPU)
            noisy_patches = noisy_patches.to(device)
            target_patches = target_patches.to(device)

            optimizer.zero_grad() # Clear previous gradients
            outputs = model(noisy_patches) # Forward pass
            loss = criterion(outputs, target_patches) # Calculate loss
            loss.backward() # Backward pass (compute gradients)
            optimizer.step() # Update model parameters

            running_loss += loss.item()

            # Print training progress periodically
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Avg Loss: {running_loss / 100:.12f}")
                running_loss = 0.0 # Reset running loss for the next 100 steps
        print(f"Epoch [{epoch+1}/{epochs}] finished. Total samples processed: {(epoch+1) * len(dataloader.dataset)}")

# Run a small training session for demonstration purposes
# In a real scenario, you would train for more epochs and potentially save the model.
print("\n--- Starting Model Training (short run for demonstration) ---")
train_model(model, patch_dataloader, epochs=20, learning_rate=1e-4)
print("--- Training complete ---")

# --- 5. Inference (Sliding Window) ---
# Ensure the patch_size here matches the PATCH_SIZE used for training
def sliding_window_inference(model, input_volume, patch_size=PATCH_SIZE, stride=(5, 5, 5)):
    """
    Performs inference on a full 3D volume using a sliding window approach.
    Overlapping predictions are averaged to produce the final denoised volume.
    """
    model.eval() # Set model to evaluation mode (disables dropout, batchnorm updates)
    D, H, W = input_volume.shape # Original dimensions of the input volume
    patch_d, patch_h, patch_w = patch_size
    stride_d, stride_h, stride_w = stride

    # Pad the input volume to ensure all parts, especially edges, are covered by patches.
    # The padding ensures that the last patch can be extracted fully.
    # Calculate required padding for each dimension
    pad_d_needed = (math.ceil(D / stride_d) * stride_d) - D + (patch_d - stride_d)
    pad_h_needed = (math.ceil(H / stride_h) * stride_h) - H + (patch_h - stride_h)
    pad_w_needed = (math.ceil(W / stride_w) * stride_w) - W + (patch_w - stride_w)

    # Ensure padding is non-negative and apply symmetric padding
    pad_d_needed = max(0, pad_d_needed)
    pad_h_needed = max(0, pad_h_needed)
    pad_w_needed = max(0, pad_w_needed)

    # Pad symmetrically (half on each side)
    pad_d_before = pad_d_needed // 2
    pad_d_after = pad_d_needed - pad_d_before
    pad_h_before = pad_h_needed // 2
    pad_h_after = pad_h_needed - pad_h_before
    pad_w_before = pad_w_needed // 2
    pad_w_after = pad_w_needed - pad_w_before

    # Add batch and channel dimensions for padding and model input: [1, 1, D, H, W]
    padded_input = F.pad(input_volume.unsqueeze(0).unsqueeze(0),
                         (pad_w_before, pad_w_after,
                          pad_h_before, pad_h_after,
                          pad_d_before, pad_d_after), mode='reflect') # 'reflect' mode for boundary

    # Get the new dimensions of the padded volume
    _, _, padded_D, padded_H, padded_W = padded_input.shape

    # Initialize output volume and a weight map with the dimensions of the padded volume
    output_volume = torch.zeros((padded_D, padded_H, padded_W), dtype=torch.float32).to(device)
    weight_map = torch.zeros((padded_D, padded_H, padded_W), dtype=torch.float32).to(device)


    print(f"\nStarting sliding window inference on padded volume of shape: ({padded_D}, {padded_H}, {padded_W})")
    print(f"Patch size: {patch_size}, Stride: {stride}")

    with torch.no_grad(): # Disable gradient calculations for inference
        # Wrap the outermost loop with tqdm for a progress bar
        for d in tqdm(range(0, padded_D - patch_d + 1, stride_d), desc="Denoising Progress (Depth)"):
            for h in range(0, padded_H - patch_h + 1, stride_h):
                # Wrap the innermost loop with tqdm for a progress bar
                for w in tqdm(range(0, padded_W - patch_w + 1, stride_w), desc=f"Denoising Progress (H={h}, W)", leave=False):
                    # Extract a patch from the padded input volume
                    patch = padded_input[:, :, d:d+patch_d, h:h+patch_h, w:w+patch_w]
                    patch = patch.to(device) # Move patch to device

                    # Get prediction from the model
                    predicted_patch = model(patch)

                    # Add the predicted patch to the corresponding region in the output volume
                    # and increment the weight map to count overlaps.
                    # Squeeze to remove batch and channel dimensions for accumulation.
                    output_volume[d:d+patch_d, h:h+patch_h, w:w+patch_w] += predicted_patch.squeeze(0).squeeze(0)
                    weight_map[d:d+patch_d, h:h+patch_h, w:w+patch_w] += 1.0 # Mark this region as covered

    # Handle cases where weight_map might have zeros (should not happen with correct padding/stride)
    # Add a small epsilon to avoid division by zero if any region was somehow missed
    weight_map[weight_map == 0] = 1e-6

    # Average the overlapping predictions by dividing the accumulated output by the weight map
    denoised_volume = output_volume / weight_map

    # Crop the denoised volume back to the original dimensions
    denoised_volume = denoised_volume[pad_d_before:pad_d_before+D,
                                      pad_h_before:pad_h_before+H,
                                      pad_w_before:pad_w_before+W]

    return denoised_volume

# Perform inference on the noisy_volume after the model has been trained
print("\n--- Starting Model Inference (Sliding Window) ---")
denoised_result = sliding_window_inference(model, noisy_volume.to(device))
print(f"Denoised Volume Shape: {denoised_result.shape}")
print("--- Inference complete ---")

# --- 6. Reshape and Plot Results ---
def plot_denoising_results(noisy_3d_volume, target_3d_volume, denoised_3d_volume,
                            original_a, original_e,
                            M_indices_to_plot: list, E_indices_to_plot: list, angleRange=(0, 70), energyRange=(-0.6, 0)):
    """
    Reshapes the 3D volumes back to [M, E, a, e] and plots slices for comparison.

    Args:
        noisy_3d_volume (torch.Tensor): The original noisy 3D volume [M, E, a*e].
        target_3d_volume (torch.Tensor): The original target 3D volume [M, E, a*e].
        denoised_3d_volume (torch.Tensor): The denoised 3D volume [M, E, a*e].
        original_a (int): The original 'a' dimension before flattening.
        original_e (int): The original 'e' dimension before flattening.
        M_indices_to_plot (list): List of M indices to plot slices from.
        E_indices_to_plot (list): List of E indices to plot slices from.
        angleRange (tuple): (min_angle, max_angle) for the 'e' axis (x-axis).
        energyRange (tuple): (min_energy, max_energy) for the 'a' axis (y-axis).
    """
    # Reshape the 3D volumes back to 4D [M, E, a, e]
    M, E, _ = noisy_3d_volume.shape
    noisy_4d = noisy_3d_volume.reshape(M, E, original_a, original_e).cpu().numpy()
    target_4d = target_3d_volume.reshape(M, E, original_a, original_e).cpu().numpy()
    denoised_4d = denoised_3d_volume.reshape(M, E, original_a, original_e).cpu().numpy()

    print(f"\nReshaped volumes to 4D: [M={M}, E={E}, a={original_a}, e={original_e}]")

    # Define the extent for imshow for correct axis labeling
    # extent = [left, right, bottom, top]
    # For a slice of shape (original_a, original_e), 'e' is on x-axis, 'a' is on y-axis
    # With origin='lower', (0,0) is bottom-left.
    extent = [angleRange[0], angleRange[1], energyRange[0], energyRange[1]]

    for m_idx in M_indices_to_plot:
        for e_idx in E_indices_to_plot:
            if m_idx >= M or e_idx >= E:
                print(f"Warning: Indices M={m_idx}, E={e_idx} are out of bounds. Skipping plot.")
                continue

            print(f"Plotting slices for M={m_idx}, E={e_idx}")
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Comparison at M={m_idx}, E={e_idx} (Slice of a x e)')

            # Plot Noisy Data
            im0 = axes[0].imshow(noisy_4d[m_idx, e_idx, :, :], cmap='viridis', aspect='auto', origin='lower', extent=extent)
            axes[0].set_title(f'Noisy (M={m_idx}, E={e_idx})')
            axes[0].set_xlabel('e')
            axes[0].set_ylabel('a')
            plt.colorbar(im0, ax=axes[0])


            # Plot Target Data
            im1 = axes[1].imshow(target_4d[m_idx, e_idx, :, :], cmap='viridis', aspect='auto', origin='lower', extent=extent)
            axes[1].set_title(f'Target (M={m_idx}, E={e_idx})')
            axes[1].set_xlabel('e')
            axes[1].set_ylabel('a')
            plt.colorbar(im1, ax=axes[1])


            # Plot Denoised Data
            im2 = axes[2].imshow(denoised_4d[m_idx, e_idx, :, :], cmap='viridis', aspect='auto', origin='lower', extent=extent)
            axes[2].set_title(f'Denoised (M={m_idx}, E={e_idx})')
            axes[2].set_xlabel('e')
            axes[2].set_ylabel('a')
            plt.colorbar(im2, ax=axes[2])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            plt.savefig(f'Comparison_M{m_idx}_E{e_idx}.pdf')
            plt.close()

# Call the plotting function after inference
# Ensure ORIGINAL_A and ORIGINAL_E are set from load_and_prepare_data
if ORIGINAL_A is not None and ORIGINAL_E is not None:
    # --- Define your custom M and E indices here ---
    # # Example: First, a middle, and last index for M and E
    # M_to_plot_custom = []
    # if noisy_volume.shape[0] > 0: M_to_plot_custom.append(0)
    # if noisy_volume.shape[0] > 1: M_to_plot_custom.append(noisy_volume.shape[0] // 2)
    # if noisy_volume.shape[0] > 2: M_to_plot_custom.append(noisy_volume.shape[0] - 1)
    # M_to_plot_custom = sorted(list(set(M_to_plot_custom))) # Remove duplicates and sort

    # E_to_plot_custom = []
    # if noisy_volume.shape[1] > 0: E_to_plot_custom.append(0)
    # if noisy_volume.shape[1] > 1: E_to_plot_custom.append(noisy_volume.shape[1] // 2)
    # if noisy_volume.shape[1] > 2: E_to_plot_custom.append(noisy_volume.shape[1] - 1)
    # E_to_plot_custom = sorted(list(set(E_to_plot_custom))) # Remove duplicates and sort

    # You can manually set these lists if you want very specific indices:
    M_to_plot_custom = [0, 15, 30, 45, 50] # Example for M indices
    E_to_plot_custom = [0, 10, 25, 40, 50] # Example for E indices

    # --- Loop through all combinations of M and E indices to plot ---
    for m_idx in M_to_plot_custom:
        for e_idx in E_to_plot_custom:
            plot_denoising_results(noisy_volume, target_volume, denoised_result,
                                    original_a_dim, original_e_dim,
                                    M_indices_to_plot=[m_idx], # Pass a single M index
                                    E_indices_to_plot=[e_idx]) # Pass a single E index
else:
    print("Could not plot results: Original 'a' and 'e' dimensions were not determined.")