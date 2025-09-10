# --- 1. Imports ---
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from einops import rearrange

# Attempt to import torch.cuda.amp for mixed precision training
try:
    from torch.cuda.amp import autocast, GradScaler
    mixed_precision_available = True
    print("[INFO] PyTorch AMP (Mixed Precision) is available.")
except ImportError:
    mixed_precision_available = False
    print("[INFO] PyTorch AMP (Mixed Precision) is not available. Training will run in full precision.")


# --- 2. Global Constants ---
amplitude_scaling_factor = 1000.0

# --- 3. Data Loading Function (Refactored for memory-efficiency) ---
def load_data_and_metadata_from_npz(method, npz_path):
    """
    Loads all metadata and returns a memory-mapped object for the histograms.
    This avoids loading the full dataset into RAM.
    """
    print(f"[INFO] Loading data from: {npz_path}")
    method = method.lower()
    if method not in {"transformation", "normalization"}:
        raise ValueError(
            f"Unknown method '{method}'. Expected 'transformation' "
            f"or 'normalization'."
        )
    try:
        # Load the file to read metadata keys first
        npz = np.load(npz_path, allow_pickle=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"The file {npz_path} was not found.") from exc
    except Exception as exc:
        raise RuntimeError(f"Unable to load {npz_path}") from exc

    try:
        # Get shape without loading the whole array
        histograms_shape = npz['histograms'].shape
        if histograms_shape[0] != 2:
            raise ValueError("Expected 'histograms' data to be 5D with shape [2, M, E, a, e].")
    except KeyError as exc:
        raise KeyError("'histograms' key is missing from the .npz file") from exc

    try:
        ct_values = npz['HU'].astype(np.float32)
        initial_energies = npz['energies'].astype(np.float32)
    except KeyError as exc:
        raise KeyError("The .npz file must contain 'HU' and 'energies' keys") from exc
    
    # Get metadata from the NPZ file
    M_dim, E_dim, A_Dim, e_dim = histograms_shape[1:]

    base_keys = {"histograms", "HU", "energies"}
    extra = {k: npz[k] for k in npz.files if k not in base_keys}
    
    npz.close() # Close the initial file handle to save memory

    # Now, open the NPZ file again using a memory map ('r') for efficient slicing
    histograms_mmap = np.load(npz_path, mmap_mode='r')['histograms']
    
    return npz_path, histograms_mmap, M_dim, E_dim, A_Dim, e_dim, ct_values, initial_energies, method, extra

# --- 4. Optimized On-the-fly Dataset for NPZ files ---
class OnTheFlyNPZDataset(Dataset):
    """
    Loads data on-the-fly from a large NPZ file using a memory map to avoid
    re-opening the file and minimize I/O overhead.
    """
    def __init__(self, npz_path, histograms_mmap, ct_vals, initial_e_vals, M, E, A, e, augment_data=False, amplitude_scaling_factor=1000.0):
        self.npz_path = npz_path
        self.histograms_mmap = histograms_mmap # Store the memory-mapped object
        self.M, self.E, self.A, self.e = M, E, A, e
        self.augment_data = augment_data
        self.amplitude_scaling_factor = amplitude_scaling_factor

        self.ct_vals = torch.as_tensor(ct_vals, dtype=torch.float32)
        self.initial_e_vals = torch.as_tensor(initial_e_vals, dtype=torch.float32)

        x_coords = torch.linspace(0, 1, self.A, dtype=torch.float32)
        y_coords = torch.linspace(0, 1, self.e, dtype=torch.float32)
        self.x_grid = x_coords.view(self.A, 1).expand(self.A, self.e)
        self.y_grid = y_coords.view(1, self.e).expand(self.A, self.e)

        ct_min, ct_max = self.ct_vals.min(), self.ct_vals.max()
        e_min, e_max = self.initial_e_vals.min(), self.initial_e_vals.max()
        self.ct_scale = (self.ct_vals - ct_min) / (ct_max - ct_min + 1e-12)
        self.e_scale = (self.initial_e_vals - e_min) / (e_max - e_min + 1e-12)

    def __len__(self):
        return self.M * self.E

    def __getitem__(self, idx):
        m_idx = idx // self.E
        e_idx = idx % self.E
        
        # Access the memory-mapped histograms directly - this is the key optimization.
        # This line reads a small slice from the file, not the whole file.
        noisy_input_np = self.histograms_mmap[0, m_idx, e_idx, :, :]
        true_target_np = self.histograms_mmap[1, m_idx, e_idx, :, :]
        
        # Convert to torch tensors and scale
        hist = torch.from_numpy(noisy_input_np).float() * self.amplitude_scaling_factor
        target = torch.from_numpy(true_target_np).float() * self.amplitude_scaling_factor
        
        ct_channel = self.ct_scale[m_idx].expand(self.A, self.e)
        e_channel = self.e_scale[e_idx].expand(self.A, self.e)

        input_img = torch.stack(
            [hist, ct_channel, e_channel, self.x_grid, self.y_grid], dim=0)

        if self.augment_data:
            if torch.rand(1).item() > 0.5:
                input_img = torch.flip(input_img, dims=[1])
                target = torch.flip(target, dims=[0])
            if torch.rand(1).item() > 0.5:
                input_img = torch.flip(input_img, dims=[2])
                target = torch.flip(target, dims=[1])
        
        target = target.unsqueeze(0)

        return input_img, target, idx

# --- 5. Hybrid CNN-ViT Architecture ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(out_channels)
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        if self.identity_conv:
            identity = self.identity_conv(identity)
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += identity
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True), nn.InstanceNorm2d(f_int))
        self.W_x = nn.Sequential(nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True), nn.InstanceNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.InstanceNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# New Transformer-based Bottleneck
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x is B x N x C where N is num_tokens
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x

class DenoisingUNet_ViT(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, features_list=None, image_size=100, patch_size=2, num_layers=4):
        super(DenoisingUNet_ViT, self).__init__()
        if features_list is None:
            features_list = [32, 64, 128, 256, 512]

        self.patch_size = patch_size
        self.features_list = features_list
        
        # Calculate number of patches and transformer dimension
        h_down, w_down = image_size // (2**4), image_size // (2**4) # Image size after 4 pooling layers
        
        if (h_down % patch_size != 0) or (w_down % patch_size != 0):
            raise ValueError(f"Image size after downsampling ({h_down}x{w_down}) must be divisible by patch size ({patch_size}).")
            
        self.transformer_dim = features_list[4]
        self.num_layers = num_layers
        
        patch_dim = features_list[4] * patch_size * patch_size
        num_patches = (h_down // patch_size) * (w_down // patch_size)
        
        # --- Encoder Path (CNN) ---
        self.enc1 = ResidualBlock(in_channels, features_list[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.1)

        self.enc2 = ResidualBlock(features_list[0], features_list[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=0.2)

        self.enc3 = ResidualBlock(features_list[1], features_list[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=0.3)

        self.enc4 = ResidualBlock(features_list[2], features_list[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout2d(p=0.4)

        self.conv_bottleneck_initial = ResidualBlock(features_list[3], features_list[4])
        self.dropout_bottleneck_cnn = nn.Dropout2d(p=0.5)
        
        # --- Transformer Bottleneck ---
        # 1. Patching and Embedding (now a standard Linear layer)
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.transformer_dim),
            nn.LayerNorm(self.transformer_dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.transformer_dim))
        
        # 2. Transformer Layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(self.transformer_dim, num_heads=8, mlp_dim=self.transformer_dim * 4)
            for _ in range(num_layers)
        ])
        
        # 3. Transformer Output to Feature Map (now a standard CNN layer)
        self.conv_bottleneck_final = ResidualBlock(self.transformer_dim, features_list[4])
        self.dropout_bottleneck_final = nn.Dropout2d(p=0.5)

        # --- Decoder Path (CNN) ---
        self.upconv1 = nn.ConvTranspose2d(features_list[4], features_list[3], kernel_size=2, stride=2)
        self.att1 = AttentionBlock(f_g=features_list[3], f_l=features_list[3], f_int=features_list[3]//2)
        self.dec1 = ResidualBlock(features_list[3] + features_list[3], features_list[3])
        self.dropout5 = nn.Dropout2d(p=0.4)

        self.upconv2 = nn.ConvTranspose2d(features_list[3], features_list[2], kernel_size=2, stride=2)
        self.att2 = AttentionBlock(f_g=features_list[2], f_l=features_list[2], f_int=features_list[2]//2)
        self.dec2 = ResidualBlock(features_list[2] + features_list[2], features_list[2])
        self.dropout6 = nn.Dropout2d(p=0.3)

        self.upconv3 = nn.ConvTranspose2d(features_list[2], features_list[1], kernel_size=2, stride=2)
        self.att3 = AttentionBlock(f_g=features_list[1], f_l=features_list[1], f_int=features_list[1]//2)
        self.dec3 = ResidualBlock(features_list[1] + features_list[1], features_list[1])
        self.dropout7 = nn.Dropout2d(p=0.2)

        self.upconv4 = nn.ConvTranspose2d(features_list[1], features_list[0], kernel_size=2, stride=2)
        self.att4 = AttentionBlock(f_g=features_list[0], f_l=features_list[0], f_int=features_list[0]//2)
        self.dec4 = ResidualBlock(features_list[0] + features_list[0], features_list[0])
        self.dropout8 = nn.Dropout2d(p=0.1)

        self.final_conv = nn.Conv2d(features_list[0], out_channels, kernel_size=1)
        # Note: The output activation is removed here. The model returns raw logits.

    def forward(self, x):
        # --- Encoder Path ---
        enc1_out = self.enc1(x)
        pool1_out = self.pool1(enc1_out)
        pool1_out = self.dropout1(pool1_out)

        enc2_out = self.enc2(pool1_out)
        pool2_out = self.pool2(enc2_out)
        pool2_out = self.dropout2(pool2_out)

        enc3_out = self.enc3(pool2_out)
        pool3_out = self.pool3(enc3_out)
        pool3_out = self.dropout3(pool3_out)

        enc4_out = self.enc4(pool3_out)
        pool4_out = self.pool4(enc4_out)
        pool4_out = self.dropout4(pool4_out)

        # --- CNN Bottleneck ---
        bottleneck_cnn_out = self.conv_bottleneck_initial(pool4_out)
        
        # --- Transformer Bottleneck ---
        # 1. Reshape and Patch (using the einops FUNCTION)
        # Get dimensions for reshaping
        B, C, H, W = bottleneck_cnn_out.shape
        # Rearrange to tokens: [B, C, H, W] -> [B, num_patches, patch_dim]
        tokens = rearrange(bottleneck_cnn_out, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size)
        
        # 2. Embedding and Position Encoding
        tokens = self.to_patch_embedding(tokens)
        tokens += self.pos_embedding
        
        # 3. Pass through Transformer layers
        for attn_block in self.transformer_layers:
            tokens = attn_block(tokens)
            
        # 4. Reshape back to feature map (using the einops FUNCTION)
        # Rearrange to feature map: [B, num_patches, transformer_dim] -> [B, transformer_dim, h, w]
        bottleneck_trans_out = rearrange(tokens, 'b (h w) c -> b c h w', h = H // self.patch_size, w = W // self.patch_size)
        
        # 5. Final CNN block in bottleneck
        bottleneck_final_out = self.conv_bottleneck_final(bottleneck_trans_out)
        bottleneck_out = self.dropout_bottleneck_final(bottleneck_final_out)

        # --- Decoder Path ---
        # Stage 1
        up1 = self.upconv1(bottleneck_out)
        if up1.shape[2:] != enc4_out.shape[2:]:
            up1 = F.interpolate(up1, size=enc4_out.shape[2:], mode='nearest')
        att1_out = self.att1(g=up1, x=enc4_out)
        dec1_in = torch.cat([up1, att1_out], dim=1)
        dec1_out = self.dec1(dec1_in)
        dec1_out = self.dropout5(dec1_out)

        # Stage 2
        up2 = self.upconv2(dec1_out)
        if up2.shape[2:] != enc3_out.shape[2:]:
            up2 = F.interpolate(up2, size=enc3_out.shape[2:], mode='nearest')
        att2_out = self.att2(g=up2, x=enc3_out)
        dec2_in = torch.cat([up2, att2_out], dim=1)
        dec2_out = self.dec2(dec2_in)
        dec2_out = self.dropout6(dec2_out)

        # Stage 3
        up3 = self.upconv3(dec2_out)
        if up3.shape[2:] != enc2_out.shape[2:]:
            up3 = F.interpolate(up3, size=enc2_out.shape[2:], mode='nearest')
        att3_out = self.att3(g=up3, x=enc2_out)
        dec3_in = torch.cat([up3, att3_out], dim=1)
        dec3_out = self.dec3(dec3_in)
        dec3_out = self.dropout7(dec3_out)

        # Stage 4
        up4 = self.upconv4(dec3_out)
        if up4.shape[2:] != enc1_out.shape[2:]:
            up4 = F.interpolate(up4, size=enc1_out.shape[2:], mode='nearest')
        att4_out = self.att4(g=up4, x=enc1_out)
        dec4_in = torch.cat([up4, att4_out], dim=1)
        dec4_out = self.dec4(dec4_in)
        dec4_out = self.dropout8(dec4_out)

        # --- Final Layer ---
        logits = self.final_conv(dec4_out)
        return logits
    
# --- 6. Custom Zero-Aware Hybrid Loss ---
class DownsampledMSELoss(nn.Module):
    """
    A loss function that calculates the mean squared error (MSE) between
    inputs and targets at multiple downsampled resolutions.
    """
    def __init__(self, downsample_factors=None):
        super(DownsampledMSELoss, self).__init__()
        self.downsample_factors = downsample_factors if downsample_factors is not None else []
        self.mse_loss = nn.MSELoss()
        print(f"[INFO] Initializing DownsampledMSELoss with downsample factors: {self.downsample_factors}")
        
    def forward(self, inputs, targets):
        
        # Calculate and sum the losses for downsampled resolutions ONLY
        total_loss = 0.0
        
        for factor in self.downsample_factors:
            # Downsample using average pooling to create lower-resolution histograms
            downsampled_inputs = F.avg_pool2d(inputs, kernel_size=factor, stride=factor)
            downsampled_targets = F.avg_pool2d(targets, kernel_size=factor, stride=factor)
            
            total_loss += self.mse_loss(downsampled_inputs, downsampled_targets)

        return total_loss
    
# --- 7. Denoising Model Class ---
class DenoisingModel:
    def __init__(self, model_params, training_params, image_size=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # The model is now DenoisingUNet_ViT
        self.model = DenoisingUNet_ViT(
            in_channels=model_params['in_channels'],
            out_channels=model_params['out_channels'],
            features_list=model_params['features_list'],
            image_size=image_size
        ).to(self.device)

        self.criterion = DownsampledMSELoss(
            downsample_factors=training_params['downsample_factors'] # Pass the new parameter here
        )
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                     lr=training_params['learning_rate'], 
                                     weight_decay=training_params['weight_decay'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                           mode='min', 
                                           factor=training_params['scheduler_factor'], 
                                           patience=training_params['scheduler_patience'])
        # Initialize a GradScaler for mixed precision training
        if mixed_precision_available and self.device.type == 'cuda':
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def train_model(self, train_loader, val_loader, num_epochs, early_stopping_patience=10):
        print("[INFO] Starting model training...")
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_wts = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                
                if self.scaler:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)

            self.model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for inputs, targets, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VALID]"):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_running_loss += loss.item() * inputs.size(0)

            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            val_losses.append(val_epoch_loss)

            self.scheduler.step(val_epoch_loss)

            if val_epoch_loss <= best_val_loss:
                best_val_loss = val_epoch_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
                print(f'\nEpoch {epoch+1}/{num_epochs}, NEW BEST VAL LOSS: {val_epoch_loss:.4f}. Model saved.')
                torch.save(best_model_wts, 'best_model_weights.pth')
            else:
                epochs_without_improvement += 1
                print(f'\nEpoch {epoch+1}/{num_epochs}, No Improvement. Epochs without improvement: {epochs_without_improvement}')

            if epochs_without_improvement >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1} as validation loss has not improved for {early_stopping_patience} consecutive epochs.')
                break

            print(f'Current Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')

        print("Model training finished.")
        print("-" * 30)

        if best_model_wts:
            self.model.load_state_dict(best_model_wts)
        else:
            print("Warning: No model weights were saved as validation loss never improved.")

        return train_losses, val_losses

# --- 8. Functions for Denoising, Plotting, and Saving ---
def denoise_full_dataset(denoising_model, histograms_mmap, npz_path, original_shape, hu_values, initial_energies, batch_size=128):
    """
    Denoises the entire dataset and returns the result as a 4D numpy array.
    """
    print("\nStarting full dataset denoising...")
    
    M, E, A, e = original_shape
    
    full_dataset = OnTheFlyNPZDataset(npz_path, histograms_mmap, hu_values, initial_energies, M, E, A, e, 
                                     augment_data=False, amplitude_scaling_factor=amplitude_scaling_factor)

    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    
    denoised_outputs_list = []
    
    denoising_model.model.eval()
    with torch.no_grad():
        for inputs, _, _ in tqdm(full_loader, desc="Denoising batches"):
            inputs = inputs.to(denoising_model.device)
            raw_denoised = denoising_model.model(inputs)
            
            raw_denoised_np = raw_denoised.cpu().numpy()
            noisy_inputs_np = inputs.cpu().numpy()

            corrected_batch_denoised = np.zeros_like(raw_denoised_np)

            for i in range(raw_denoised_np.shape[0]):
                noisy_slice_scaled = noisy_inputs_np[i, 0, :, :]
                denoised_slice = raw_denoised_np[i, 0, :, :]

                # Check for zero-state condition based on the noisy input
                if np.sum(noisy_slice_scaled) < 1e-12:
                    # If the noisy input is zero, the output should also be zero
                    corrected_batch_denoised[i, 0, :, :] = np.zeros_like(denoised_slice)
                else:
                    # Apply softmax to enforce a sum-to-one probability distribution
                    # This is the post-processing step for the raw logits from the model
                    flat_denoised = denoised_slice.flatten()
                    softmax_output = np.exp(flat_denoised) / np.sum(np.exp(flat_denoised))
                    corrected_batch_denoised[i, 0, :, :] = softmax_output.reshape(denoised_slice.shape)
            
            denoised_outputs_list.append(corrected_batch_denoised)

    all_denoised = np.concatenate(denoised_outputs_list, axis=0)
    denoised_4d = all_denoised.reshape(original_shape)
    
    # Calculate sum of noisy1, denoise and noisy2
    print(f'[INFO] Total sum of the three 4D arrays ...')
    noisy_sum_4d = np.sum(histograms_mmap[0])
    denoised_sum_4d = np.sum(denoised_4d)
    reference_sum_4d = np.sum(histograms_mmap[1])
    
    print(f'\t Noisy sum: {noisy_sum_4d}')
    print(f'\t Denoised sum: {denoised_sum_4d}')
    print(f'\t Reference sum: {reference_sum_4d}')

    return denoised_4d

def save_denoised_npz(denoised_4d, output_filename, hu_values, initial_energies, method, extra=None):
    """
    Saves the denoised histogram and metadata to an NPZ file.
    """
    output_filename = f"{output_filename}_{method}.npz"
    print(f"\nSaving denoised data to {output_filename}...")
    
    payload = {"probTable": denoised_4d, "HU": hu_values, "energies": initial_energies}

    if method == "transformation":
        if extra is None or "rho" not in extra:
            raise ValueError("For 'transformation' method, 'extra' with 'rho' key must be provided.")
        payload["rho"] = extra["rho"]
    elif method == "normalization":
        if extra is None:
            raise ValueError("For 'normalization' method, 'extra' with 5 keys must be provided.")
        keys = ["rho", "thetaMax", "thetaMin", "energyMin", "energyMax"]
        payload.update({k: extra[k] for k in keys})
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'transformation' or 'normalization'.")
    
    np.savez(output_filename, **payload)
    print(f"Denoised 4-D data and metadata written to {output_filename}")


def plot_denoising_results_from_npz(histograms_mmap, denoised_4d, hu_values, initial_energies, method, plot_indices, directory="./plots"):
    """
    Generates plots comparing noisy, denoised, and true histograms from the saved data.
    """
    print("\nGenerating denoising plots...")
    M, E, A, e = original_4d_shape
    
    # Determine axes ranges based on the method
    x_angles_range = np.linspace(0, 1, A) if method == 'normalization' else np.linspace(0, 70, A)
    y_final_energies_range = np.linspace(0, 1, e) if method == 'normalization' else np.linspace(-0.6, 0, e)
    extent = [np.min(x_angles_range), np.max(x_angles_range), np.min(y_final_energies_range), np.max(y_final_energies_range)]

    if not os.path.exists(directory):
        os.makedirs(directory)

    for idx in plot_indices:
        m_idx_to_plot = idx // E
        e_idx_to_plot = idx % E

        noisy_input_np = histograms_mmap[0, m_idx_to_plot, e_idx_to_plot, :, :]
        true_clean_np = histograms_mmap[1, m_idx_to_plot, e_idx_to_plot, :, :]
        denoised_output_np = denoised_4d[m_idx_to_plot, e_idx_to_plot, :, :]
        
        # Convert to dB
        noisy_input_db = 10 * np.log10(noisy_input_np + 1e-12)
        denoised_output_db = 10 * np.log10(denoised_output_np + 1e-12)
        true_clean_db = 10 * np.log10(true_clean_np + 1e-12)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Denoising Example (M: {hu_values[m_idx_to_plot]:.0f} CT, E: {initial_energies[e_idx_to_plot]:.0f} keV)', fontsize=16)

        im1 = axes[0].imshow(noisy_input_db.T, origin='lower', aspect='auto', cmap='Reds', extent=extent)
        axes[0].set_title('Noisy Input 1')
        plt.colorbar(im1, ax=axes[0], label='Probability (dB)')
        if method == 'normalization':
            axes[0].set_xlabel('Normalized Angle (a.u.)')
            axes[0].set_ylabel('Normalized Energy (a.u.)')
        else:
            axes[0].set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
            axes[0].set_ylabel(r'$\frac{ln((E_i-E_f)/E_i)}{\sqrt{E_i}}$ (MeV$^{-1/2}$)')

        im2 = axes[1].imshow(denoised_output_db.T, origin='lower', aspect='auto', cmap='Reds', extent=extent)
        axes[1].set_title('Predicted Denoised')
        plt.colorbar(im2, ax=axes[1], label='Probability (dB)')
        if method == 'normalization':
            axes[1].set_xlabel('Normalized Angle (a.u.)')
            axes[1].set_ylabel('Normalized Energy (a.u.)')
        else:
            axes[1].set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
            axes[1].set_ylabel(r'$\frac{ln((E_i-E_f)/E_i)}{\sqrt{E_i}}$ (MeV$^{-1/2}$)')

        im3 = axes[2].imshow(true_clean_db.T, origin='lower', aspect='auto', cmap='Reds', extent=extent)
        axes[2].set_title('Noisy Input 2')
        plt.colorbar(im3, ax=axes[2], label='Probability (dB)')
        if method == 'normalization':
            axes[2].set_xlabel('Normalized Angle (a.u.)')
            axes[2].set_ylabel('Normalized Energy (a.u.)')
        else:
            axes[2].set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
            axes[2].set_ylabel(r'$\frac{ln((E_i-E_f)/E_i)}{\sqrt{E_i}}$ (MeV$^{-1/2}$)')

        plt.tight_layout()
        plt.savefig(f'{directory}/denoising_example_full_run_M{m_idx_to_plot}_E{e_idx_to_plot}.pdf')
        plt.close(fig)
    print(f"Generated {len(plot_indices)} denoising plots.")


# --- 9. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a denoising CNN with real data from an NPZ file.")
    data_source_group = parser.add_mutually_exclusive_group(required=True)
    data_source_group.add_argument('--transformation', action='store_true', help="Use transformation method")
    data_source_group.add_argument('--normalization', action='store_true', help="Use normalization method")
    parser.add_argument("--npz", type=str, default=None, help="Optional path to a custom NPZ file.")
    args = parser.parse_args()
    
    # 1. Data Loading and Initial Setup
    method = 'transformation' if args.transformation else 'normalization'
    npz_path = args.npz if args.npz else (
        './DenoisingDataTransSheet.npz' if method == 'transformation' else './DenoisingDataNormSheet.npz'
    )
    
    npz_path, histograms_mmap, num_materials, num_initial_energies, num_angles, num_final_energies, ct_values, initial_energies, method, extra = load_data_and_metadata_from_npz(method=method, npz_path=npz_path)

    original_4d_shape = (num_materials, num_initial_energies, num_angles, num_final_energies)
    print(f"[INFO] Original 4D shape: {original_4d_shape}")
    
    # Check if the image dimensions are 100x100
    if num_angles != 100 or num_final_energies != 100:
        raise ValueError("The provided NPZ data does not have a 100x100 histogram size.")

    # 2. Dataset and DataLoader
    full_dataset = OnTheFlyNPZDataset(
        npz_path=npz_path,
        histograms_mmap=histograms_mmap,
        ct_vals=ct_values,
        initial_e_vals=initial_energies,
        M=num_materials, E=num_initial_energies, A=num_angles, e=num_final_energies,
        augment_data=True,
        amplitude_scaling_factor=amplitude_scaling_factor
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"[INFO] Number of training samples: {len(train_dataset)}")
    print(f"[INFO] Number of validation samples: {len(val_dataset)}")
    
    # 3. Model Parameters and Training
    model_params = {
        'in_channels': 5,
        'out_channels': 1,
        # features_list is now larger for a more powerful ViT-Hybrid model
        'features_list': [8, 16, 32, 64, 128], 
    }
    
    print(f'[INFO] Model Parameters:')
    print(f'\t.In Channels: {model_params["in_channels"]}')
    print(f'\t.Out Channels: {model_params["out_channels"]}')
    print(f'\t.Features List: {model_params["features_list"]}')

    training_params = {
        'learning_rate': 5e-5,
        'weight_decay': 1e-5,
        'scheduler_factor': 0.5,
        'scheduler_patience': 5,
        'downsample_factors': [2, 4] # NEW: Multi-resolution training
    }
    
    print(f'[INFO] Training Parameters:')
    print(f'\t.Learning Rate: {training_params["learning_rate"]}')
    print(f'\t.Weight Decay: {training_params["weight_decay"]}')
    print(f'\t.Scheduler Factor: {training_params["scheduler_factor"]}')
    print(f'\t.Scheduler Patience: {training_params["scheduler_patience"]}')
    print(f'\t.Downsample Factors: {training_params["downsample_factors"]}')

    denoising_model = DenoisingModel(model_params, training_params, image_size=100)
    train_losses, val_losses = denoising_model.train_model(
        train_loader, val_loader, num_epochs=70
    )
    print('[INFO] Number of parameters in the model:', sum(p.numel() for p in denoising_model.model.parameters() if p.requires_grad))

    # 4. Denoising the full dataset and plotting
    denoised_4d_data = denoise_full_dataset(
        denoising_model, histograms_mmap, npz_path, original_4d_shape, ct_values, initial_energies
    )

    # Save the denoised data
    save_denoised_npz(denoised_4d_data, "denoised_output_advanced", ct_values, initial_energies, method, extra)
    
    # Plotting example results
    num_materials_to_plot = min(num_materials, 10)
    num_initial_energies_to_plot = min(num_initial_energies, 10)
    m_indices_to_plot = np.linspace(0, num_materials - 1, num_materials_to_plot, dtype=int)
    e_indices_to_plot = np.linspace(0, num_initial_energies - 1, num_initial_energies_to_plot, dtype=int)
    
    plot_indices = []
    for m in m_indices_to_plot:
        for e in e_indices_to_plot:
            plot_indices.append(m * num_initial_energies + e)
    
    plot_denoising_results_from_npz(
        histograms_mmap, denoised_4d_data, ct_values, initial_energies, method, plot_indices
    )
    
    # Print keys in the denoised histograms
    output_filename = f"denoised_output_advanced_{method}.npz"
    histograms = np.load(output_filename, allow_pickle=True)
    for key in histograms.keys():
        print(f"Key: {key}, Shape: {histograms[key].shape}")