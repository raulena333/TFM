
import torch
import torch.nn as nn

# --- 1. Imports ---
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import copy
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Attempt to import torch.cuda.amp for mixed precision training
try:
    from torch.cuda.amp import autocast, GradScaler
    mixed_precision_available = True
    print("[INFO] PyTorch AMP (Mixed Precision) is available.")
except ImportError:
    mixed_precision_available = False
    print("[INFO] PyTorch AMP (Mixed Precision) is not available. Training will run in full precision.")

# --- 2. Global Constants ---
amplitude_scaling_factor = 1.0

# --- 3. Data Loading Function (Refactored for memory-efficiency) ---
def load_data_and_metadata_from_npz(method, npz_path):
    """
    Loads all metadata and returns a memory-mapped object for the histograms.
    This avoids loading the full dataset into RAM, which is crucial for large files.
    """
    print(f"[INFO] Loading data from: {npz_path}")
    method = method.lower()
    if method not in {"transformation", "normalization"}:
        raise ValueError(
            f"Unknown method '{method}'. Expected 'transformation' "
            f"or 'normalization'."
        )
    try:
        npz = np.load(npz_path, allow_pickle=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"The file {npz_path} was not found.") from exc
    except Exception as exc:
        raise RuntimeError(f"Unable to load {npz_path}") from exc
    try:
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
    M_dim, E_dim, A_Dim, e_dim = histograms_shape[1:]
    base_keys = {"histograms", "HU", "energies"}
    extra = {k: npz[k] for k in npz.files if k not in base_keys}
    npz.close()
    histograms_mmap = np.load(npz_path, mmap_mode='r')['histograms']
    return npz_path, histograms_mmap, M_dim, E_dim, A_Dim, e_dim, ct_values, initial_energies, method, extra

# --- 4. Optimized On-the-fly Dataset for NPZ files ---
class OnTheFlyNPZDataset(Dataset):
    """
    Custom PyTorch Dataset that loads data samples on-the-fly from a large NPZ file
    using a memory map. This is highly memory-efficient.
    """
    def __init__(self, npz_path, histograms_mmap, ct_vals, initial_e_vals, M, E, A, e, augment_data=False, amplitude_scaling_factor=1000.0):
        self.npz_path = npz_path
        self.histograms_mmap = histograms_mmap
        self.M, self.E, self.A, self.e = M, E, A, e
        self.augment_data = augment_data
        self.amplitude_scaling_factor = amplitude_scaling_factor

        self.ct_vals = torch.as_tensor(ct_vals, dtype=torch.float32)
        self.initial_e_vals = torch.as_tensor(initial_e_vals, dtype=torch.float32)

        # Create grid channels for the U-Net input
        x_coords = torch.linspace(0, 1, self.A, dtype=torch.float32)
        y_coords = torch.linspace(0, 1, self.e, dtype=torch.float32)
        self.x_grid = x_coords.view(self.A, 1).expand(self.A, self.e)
        self.y_grid = y_coords.view(1, self.e).expand(self.A, self.e)

        # Scale metadata values to a 0-1 range
        ct_min, ct_max = self.ct_vals.min(), self.ct_vals.max()
        e_min, e_max = self.initial_e_vals.min(), self.initial_e_vals.max()
        self.ct_scale = (self.ct_vals - ct_min) / (ct_max - ct_min + 1e-12)
        self.e_scale = (self.initial_e_vals - e_min) / (e_max - e_min + 1e-12)

    def __len__(self):
        return self.M * self.E

    def __getitem__(self, idx):
        m_idx = idx // self.E
        e_idx = idx % self.E
        
        # Load data from memory map, which is very fast
        noisy_input_np = self.histograms_mmap[0, m_idx, e_idx, :, :]
        true_target_np = self.histograms_mmap[1, m_idx, e_idx, :, :]
        
        # Convert to PyTorch tensors and apply amplitude scaling
        hist = torch.from_numpy(noisy_input_np).float() * self.amplitude_scaling_factor
        target = torch.from_numpy(true_target_np).float() * self.amplitude_scaling_factor
        
        # Expand metadata and coordinate channels to match histogram size
        ct_channel = self.ct_scale[m_idx].expand(self.A, self.e)
        e_channel = self.e_scale[e_idx].expand(self.A, self.e)

        # Stack all input channels (noisy histogram, CT, energy, x-grid, y-grid)
        input_img = torch.stack(
            [hist, ct_channel, e_channel, self.x_grid, self.y_grid], dim=0)

        # Apply data augmentation if enabled
        if self.augment_data:
            if torch.rand(1).item() > 0.5:
                input_img = torch.flip(input_img, dims=[1])
                target = torch.flip(target, dims=[0])
            if torch.rand(1).item() > 0.5:
                input_img = torch.flip(input_img, dims=[2])
                target = torch.flip(target, dims=[1])
        
        target = target.unsqueeze(0)

        return input_img, target, idx

# --- 5. Robust U-Net Architecture ---
class UpsampleBlock(nn.Module):
    """
    An upsampling block that replaces ConvTranspose2d to avoid checkerboard artifacts.
    It uses nearest-neighbor interpolation followed by a standard convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(UpsampleBlock, self).__init__()
        # NOTE: We do NOT upsample here, as that is now handled in the forward pass
        # This block simply performs a convolution to refine features.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    """A standard Residual Block for the U-Net, adding skip connections to prevent vanishing gradients."""
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
    """Attention Gate for the U-Net, which helps the model focus on important features."""
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

class SelfAttentionBottleneck(nn.Module):
    """
    A bottleneck module that uses a Multi-Head Self-Attention mechanism
    to capture long-range dependencies in the feature map, which can
    be helpful for complex patterns in the histogram.
    """
    def __init__(self, in_channels, heads=8):
        super(SelfAttentionBottleneck, self).__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=heads,
            batch_first=True
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        x_norm = self.norm(x_flat)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        attn_output_reshaped = attn_output.permute(0, 2, 1).view(B, C, H, W)
        out = x + attn_output_reshaped
        return out

# --- 3. Rewritten Denoising U-Net with UpsampleBlock and Interpolation ---
class DenoisingUNet(nn.Module):
    """
    The main Denoising U-Net model with residual blocks, attention gates, and
    a self-attention bottleneck for enhanced feature learning.
    """
    def __init__(self, in_channels=5, out_channels=1, features_list=None):
        super(DenoisingUNet, self).__init__()
        if features_list is None:
            features_list = [32, 64, 128, 256, 512]
        
        # Encoder path
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
        
        # Bottleneck with a residual block and self-attention
        self.conv_bottleneck_initial = ResidualBlock(features_list[3], features_list[4])
        self.bottleneck_attn = SelfAttentionBottleneck(in_channels=features_list[4])
        self.dropout_bottleneck = nn.Dropout2d(p=0.5)

        # Decoder path - NOW USING UpsampleBlock INSTEAD OF CONVTRANSPOSE2D
        # We perform the upsampling in the forward pass to correctly handle odd dimensions
        self.upconv1 = UpsampleBlock(features_list[4], features_list[3])
        self.att1 = AttentionBlock(f_g=features_list[3], f_l=features_list[3], f_int=features_list[3]//2)
        self.dec1 = ResidualBlock(features_list[3] + features_list[3], features_list[3])
        self.dropout5 = nn.Dropout2d(p=0.4)
        
        self.upconv2 = UpsampleBlock(features_list[3], features_list[2])
        self.att2 = AttentionBlock(f_g=features_list[2], f_l=features_list[2], f_int=features_list[2]//2)
        self.dec2 = ResidualBlock(features_list[2] + features_list[2], features_list[2])
        self.dropout6 = nn.Dropout2d(p=0.3)
        
        self.upconv3 = UpsampleBlock(features_list[2], features_list[1])
        self.att3 = AttentionBlock(f_g=features_list[1], f_l=features_list[1], f_int=features_list[1]//2)
        self.dec3 = ResidualBlock(features_list[1] + features_list[1], features_list[1])
        self.dropout7 = nn.Dropout2d(p=0.2)
        
        self.upconv4 = UpsampleBlock(features_list[1], features_list[0])
        self.att4 = AttentionBlock(f_g=features_list[0], f_l=features_list[0], f_int=features_list[0]//2)
        self.dec4 = ResidualBlock(features_list[0] + features_list[0], features_list[0])
        self.dropout8 = nn.Dropout2d(p=0.1)
        
        # Final output convolution
        self.final_conv = nn.Conv2d(features_list[0], out_channels, kernel_size=1)
        self.final_activation = nn.ReLU()
    
    def forward(self, x):
        # Encoder pass
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
        
        # Bottleneck pass
        bottleneck_conv_out = self.conv_bottleneck_initial(pool4_out)
        bottleneck_attn_out = self.bottleneck_attn(bottleneck_conv_out)
        bottleneck_out = self.dropout_bottleneck(bottleneck_attn_out)

        # Decoder pass with attention gates and skip connections
        up1_interp = F.interpolate(bottleneck_out, size=enc4_out.shape[2:], mode='nearest')
        up1 = self.upconv1(up1_interp)
        att1_out = self.att1(g=up1, x=enc4_out)
        dec1_in = torch.cat([up1, att1_out], dim=1)
        dec1_out = self.dec1(dec1_in)
        dec1_out = self.dropout5(dec1_out)
        
        up2_interp = F.interpolate(dec1_out, size=enc3_out.shape[2:], mode='nearest')
        up2 = self.upconv2(up2_interp)
        att2_out = self.att2(g=up2, x=enc3_out)
        dec2_in = torch.cat([up2, att2_out], dim=1)
        dec2_out = self.dec2(dec2_in)
        dec2_out = self.dropout6(dec2_out)
        
        up3_interp = F.interpolate(dec2_out, size=enc2_out.shape[2:], mode='nearest')
        up3 = self.upconv3(up3_interp)
        att3_out = self.att3(g=up3, x=enc2_out)
        dec3_in = torch.cat([up3, att3_out], dim=1)
        dec3_out = self.dec3(dec3_in)
        dec3_out = self.dropout7(dec3_out)
        
        up4_interp = F.interpolate(dec3_out, size=enc1_out.shape[2:], mode='nearest')
        up4 = self.upconv4(up4_interp)
        att4_out = self.att4(g=up4, x=enc1_out)
        dec4_in = torch.cat([up4, att4_out], dim=1)
        dec4_out = self.dec4(dec4_in)
        dec4_out = self.dropout8(dec4_out)
        
        logits = self.final_conv(dec4_out)
        output = self.final_activation(logits)
        
        return output
    
# --- 6. Loss Function ---
class HybridDenoisingLoss(nn.Module):
    """
    A hybrid loss function combining KLDivLoss for probabilistic shape learning
    and L1Loss (Mean Absolute Error) for magnitude and sparsity correction,
    with added support for multiscale loss calculation.

    This loss is particularly effective for low-statistic, sparse data, as it
    encourages the model to learn both the correct overall shape of the distribution
    and to drive small, noisy values towards zero. The multiscale component
    improves learning by considering features at various resolutions.
    """
    def __init__(self, kl_weight=1.0, l1_weight=1.0, downsample_factors=None, downsample_weights=None):
        """
        Initializes the hybrid loss with customizable weighting factors and multiscale options.

        Args:
            kl_weight (float): The weighting factor for the KL divergence loss.
                               This component focuses on matching the probability
                               distribution shape.
            l1_weight (float): The weighting factor for the L1 loss.
                               This component encourages sparsity and corrects
                               for the magnitude of the predicted values.
            downsample_factors (list of int, optional): A list of downsampling factors.
                                                        For each factor, the loss will
                                                        be calculated on a downsampled
                                                        version of the inputs.
            downsample_weights (list of float, optional): A list of weights corresponding
                                                          to each downsample factor.
                                                          Must have the same length as
                                                          `downsample_factors`.
        """
        super(HybridDenoisingLoss, self).__init__()
        
        # PyTorch's KLDivLoss expects log-probabilities for the predictions
        # and probabilities for the targets. The `reduction='batchmean'`
        # averages the loss over all histograms in the batch.
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        # PyTorch's L1Loss calculates the mean absolute error. It's a simple,
        # but effective way to push noisy, near-zero values closer to zero.
        self.l1_loss = nn.L1Loss(reduction='mean')
        
        self.kl_weight = kl_weight
        self.l1_weight = l1_weight
        
        if downsample_factors is not None and downsample_weights is not None:
            if len(downsample_factors) != len(downsample_weights):
                raise ValueError("downsample_factors and downsample_weights must have the same length.")
        
        self.downsample_factors = downsample_factors
        self.downsample_weights = downsample_weights

        print(f"[INFO] Initializing HybridDenoisingLoss with KL_weight={self.kl_weight} and L1_weight={self.l1_weight}")
        print("[INFO] This loss is designed for low-statistic, sparse data.")
        if self.downsample_factors:
            print(f"[INFO] Multiscale loss enabled with factors: {self.downsample_factors}")

    def forward(self, inputs, targets):
        """
        Calculates the total hybrid loss.

        Args:
            inputs (torch.Tensor): Raw model outputs (logits), shape (B, 1, H, W).
            targets (torch.Tensor): True targets (probabilities), shape (B, 1, H, W).
            
        Returns:
            torch.Tensor: The calculated total hybrid loss.
        """
        # --- 1. Calculate KL Divergence Loss and L1 Loss at the original scale ---
        B, C, H, W = inputs.shape
        inputs_reshaped = inputs.view(B, -1)
        
        log_outputs = F.log_softmax(inputs_reshaped, dim=1)
        log_outputs = log_outputs.view(B, C, H, W)
        kl_loss_val = self.kl_loss(log_outputs, targets)
        
        outputs_reshaped = F.softmax(inputs_reshaped, dim=1)
        outputs_prob = outputs_reshaped.view(B, C, H, W)
        l1_loss_val = self.l1_loss(outputs_prob, targets)
        
        total_loss = (self.kl_weight * kl_loss_val) + (self.l1_weight * l1_loss_val)

        # --- 2. Calculate and add multiscale loss components ---
        if self.downsample_factors and self.downsample_weights:
            for factor, weight in zip(self.downsample_factors, self.downsample_weights):
                # Downsample inputs and targets
                downsampled_inputs = F.avg_pool2d(inputs, kernel_size=factor, stride=factor)
                downsampled_targets = F.avg_pool2d(targets, kernel_size=factor, stride=factor)
                
                # Reshape for softmax
                B, C, H_ds, W_ds = downsampled_inputs.shape
                downsampled_inputs_reshaped = downsampled_inputs.view(B, -1)

                # Calculate KL loss at the current scale
                log_outputs_ds = F.log_softmax(downsampled_inputs_reshaped, dim=1)
                log_outputs_ds = log_outputs_ds.view(B, C, H_ds, W_ds)
                kl_loss_ds = self.kl_loss(log_outputs_ds, downsampled_targets)
                
                # Calculate L1 loss at the current scale
                outputs_prob_ds = F.softmax(downsampled_inputs_reshaped, dim=1)
                outputs_prob_ds = outputs_prob_ds.view(B, C, H_ds, W_ds)
                l1_loss_ds = self.l1_loss(outputs_prob_ds, downsampled_targets)

                # Add weighted downsampled loss to the total
                total_loss += weight * ((self.kl_weight * kl_loss_ds) + (self.l1_weight * l1_loss_ds))

        return total_loss
    
    
# --- 7. Denoising Model Class ---
class DenoisingModel:
    """
    A class that encapsulates the U-Net model, the optimizer,
    and the training loop logic.
    """
    def __init__(self, model_params, training_params):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Instantiate the main U-Net model
        self.model = DenoisingUNet(
            in_channels=model_params['in_channels'],
            out_channels=model_params['out_channels'],
            features_list=model_params['features_list']
        ).to(self.device)
        
        # Initialize the Hybrid KL Divergence loss criterion
        self.criterion = HybridDenoisingLoss(
            kl_weight=training_params['kl_weight'],
            l1_weight=training_params['l1_weight'],
            downsample_factors=training_params['downsample_factors'],
            downsample_weights=training_params['downsample_weights']
        )

        self.optimizer = optim.AdamW(self.model.parameters(), 
                                     lr=training_params['learning_rate'],
                                     weight_decay=training_params['weight_decay'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                           mode='min', 
                                           factor=training_params['scheduler_factor'], 
                                           patience=training_params['scheduler_patience'])
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
    Denoises the entire dataset using the trained model and returns the denoised data.
    This corrected version applies softmax to the model's logits to get a valid
    probability distribution, consistent with the training objective. It also
    explicitly handles the edge case of all-zero noisy input.
    """
    print("\nStarting full dataset denoising...")
    M, E, A, e = original_shape
    
    # Note: 'amplitude_scaling_factor' is assumed to be defined globally or passed as an argument.
    full_dataset = OnTheFlyNPZDataset(npz_path, histograms_mmap, hu_values, initial_energies, M, E, A, e, 
                                      augment_data=False, amplitude_scaling_factor=amplitude_scaling_factor)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    denoised_outputs_list = []
    
    # Set the model to evaluation mode
    denoising_model.model.eval()

    with torch.no_grad():
        for inputs, targets, _ in tqdm(full_loader, desc="Denoising batches"):
            inputs = inputs.to(denoising_model.device)
            
            # Denoise the entire batch at once for maximum efficiency.
            raw_denoised_batch = denoising_model.model(inputs)
            
            # Apply softmax to the raw logits to get a valid probability distribution.
            B, C, H, W = raw_denoised_batch.shape
            raw_denoised_reshaped = raw_denoised_batch.view(B, -1)
            denoised_probs_reshaped = F.softmax(raw_denoised_reshaped, dim=1)
            denoised_probs = denoised_probs_reshaped.view(B, C, H, W)
            
            # Handle the zero-input edge cases
            for i in range(inputs.shape[0]):
                noisy_slice = inputs[i, 0, :, :]
                if torch.sum(noisy_slice) < 1e-12:
                    denoised_probs[i] = torch.zeros_like(denoised_probs[i])
            
            # Move the denoised probabilities to the CPU and convert to a NumPy array.
            denoised_outputs_list.append(denoised_probs.cpu().numpy())

    all_denoised = np.concatenate(denoised_outputs_list, axis=0)
    denoised_4d = all_denoised.reshape(original_shape)
    
    print(f'[INFO] Total sum of the three 4D arrays ...')
    reference_sum_4d = np.sum(histograms_mmap[1])
    noisy_sum_4d = np.sum(histograms_mmap[0])
    denoised_sum_4d = np.sum(denoised_4d)
    print(f'\t Noisy sum: {noisy_sum_4d}')
    print(f'\t Denoised sum: {denoised_sum_4d}')
    print(f'\t Reference sum: {reference_sum_4d}')
    
    return denoised_4d

def save_denoised_npz(denoised_4d, output_filename, hu_values, initial_energies, method, extra=None):
    """Saves the denoised histogram and metadata to an NPZ file."""
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

def plot_denoising_results_from_npz(histograms_mmap, denoised_4d, hu_values, initial_energies, method, plot_indices, directory="./plots", original_4d_shape=None):
    """Generates plots comparing noisy, denoised, and true histograms from the saved data."""
    print("\nGenerating denoising plots...")
    M, E, A, e = original_4d_shape
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
        noisy_input_db = 10 * np.log10(noisy_input_np + 1e-12)
        denoised_output_db = 10 * np.log10(denoised_output_np + 1e-12)
        true_clean_db = 10 * np.log10(true_clean_np + 1e-12)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Denoising Example (M: {hu_values[m_idx_to_plot]:.0f} CT, E: {initial_energies[e_idx_to_plot]:.0f} keV)', fontsize=16)
        im1 = axes[0].imshow(noisy_input_db.T, origin='lower', aspect='auto', cmap='Reds', extent=extent)
        axes[0].set_title('Noisy Input')
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
        axes[2].set_title('Reference Noisy')
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
    if num_angles != 100 or num_final_energies != 100:
        raise ValueError("The provided NPZ data does not have a 100x100 histogram size.")

    # 2. Dataset and DataLoader
    full_dataset = OnTheFlyNPZDataset(
        npz_path=npz_path,
        histograms_mmap=histograms_mmap,
        ct_vals=ct_values,
        initial_e_vals=initial_energies,
        M=num_materials, E=num_initial_energies, A=num_angles, e=num_final_energies,
        augment_data=False,
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
        'features_list': [16, 32, 64, 128, 256],
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
        'early_stopping_patience': 15,
        'num_epochs': 200,
        'kl_weight': 1.2,  
        'l1_weight': 700., 
        'downsample_factors': [2, 4, 8, 16],
        'downsample_weights': [15, 4., 0.5, 0.15],
    }
    print(f'[INFO] Training Parameters:')
    print(f'\t.Learning Rate: {training_params["learning_rate"]}')
    print(f'\t.Weight Decay: {training_params["weight_decay"]}')
    print(f'\t.Scheduler Factor: {training_params["scheduler_factor"]}')
    print(f'\t.Scheduler Patience: {training_params["scheduler_patience"]}')
    print(f'\t.Early Stopping Patience: {training_params["early_stopping_patience"]}')
    print(f'\t.Num Epochs: {training_params["num_epochs"]}')
    
    denoising_model = DenoisingModel(model_params, training_params)
    print('[INFO] Number of parameters in the U-Net:', sum(p.numel() for p in denoising_model.model.parameters() if p.requires_grad))
    # print('[INFO] Number of parameters in the Feature Extractor:', sum(p.numel() for p in denoising_model.feature_extractor.parameters() if p.requires_grad))
    
    train_losses, val_losses = denoising_model.train_model(
        train_loader, val_loader, num_epochs=training_params['num_epochs'], early_stopping_patience=training_params['early_stopping_patience']
    )

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
        histograms_mmap, denoised_4d_data, ct_values, initial_energies, method, plot_indices, directory='./plots', original_4d_shape=original_4d_shape
    )
    
    # Print keys in the denoised histograms
    output_filename = f"denoised_output_advanced_{method}.npz"
    histograms = np.load(output_filename, allow_pickle=True)
    for key in histograms.keys():
        print(f"Key: {key}, Shape: {histograms[key].shape}")