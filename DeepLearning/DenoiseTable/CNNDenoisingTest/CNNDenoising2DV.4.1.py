import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
import copy
import argparse
import os
import sys
from tqdm import tqdm

# --- 1. Global Constants ---
amplitude_scaling_factor = 1000.0

# --- 2. Data Loading Function (Refactored for memory-efficiency) ---
def load_data_and_metadata_from_npz(method, npz_path):
    """
    Loads all metadata and returns a memory-mapped object for the histograms.
    This avoids loading the full dataset into RAM.
    """
    print(f"[INFO] Loading data from: {npz_path}")
    method = method.lower()
    if method not in {"transformation", "normalization"}:
        raise ValueError(
            f"Unknown method '{method}'.  Expected 'transformation' "
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
    # THIS IS THE KEY OPTIMIZATION: `mmap_mode='r'` prevents the entire file from loading
    # into RAM. We now have a reference to the data on disk.
    histograms_mmap = np.load(npz_path, mmap_mode='r')['histograms']
    
    return npz_path, histograms_mmap, M_dim, E_dim, A_Dim, e_dim, ct_values, initial_energies, method, extra

# --- 3. Optimized On-the-fly Dataset for NPZ files ---
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
        noisy_target_np = self.histograms_mmap[1, m_idx, e_idx, :, :]
        
        # Convert to torch tensors and scale
        hist = torch.from_numpy(noisy_input_np).float() * self.amplitude_scaling_factor
        target = torch.from_numpy(noisy_target_np).float() * self.amplitude_scaling_factor
        
        mask = (hist > 1e-12).float()

        ct_channel = self.ct_scale[m_idx].expand(self.A, self.e)
        e_channel = self.e_scale[e_idx].expand(self.A, self.e)

        input_img = torch.stack(
            [hist, ct_channel, e_channel, self.x_grid, self.y_grid], dim=0)

        if self.augment_data:
            if torch.rand(1).item() > 0.5:
                input_img = torch.flip(input_img, dims=[1])
                target = torch.flip(target, dims=[0])
                mask = torch.flip(mask, dims=[0])
            if torch.rand(1).item() > 0.5:
                input_img = torch.flip(input_img, dims=[2])
                target = torch.flip(target, dims=[1])
                mask = torch.flip(mask, dims=[1])

        target = target.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return input_img, target, mask

# --- 4. Robust U-Net Architecture (Unchanged) ---
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

class DenoisingUNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, features_list=None):
        super(DenoisingUNet, self).__init__()
        if features_list is None:
            features_list = [32, 64, 128, 256]

        self.enc1 = ResidualBlock(in_channels, features_list[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.enc2 = ResidualBlock(features_list[0], features_list[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=0.3)
        self.enc3 = ResidualBlock(features_list[1], features_list[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=0.4)
        self.bottleneck = ResidualBlock(features_list[2], features_list[3])
        self.dropout_bottleneck = nn.Dropout2d(p=0.5)

        self.upconv1 = nn.ConvTranspose2d(features_list[3], features_list[2], kernel_size=2, stride=2)
        self.att1 = AttentionBlock(f_g=features_list[2], f_l=features_list[2], f_int=features_list[2]//2)
        self.dec1 = ResidualBlock(features_list[2] + features_list[2], features_list[2])
        self.dropout4 = nn.Dropout2d(p=0.4)
        self.upconv2 = nn.ConvTranspose2d(features_list[2], features_list[1], kernel_size=2, stride=2)
        self.att2 = AttentionBlock(f_g=features_list[1], f_l=features_list[1], f_int=features_list[1]//2)
        self.dec2 = ResidualBlock(features_list[1] + features_list[1], features_list[1])
        self.dropout5 = nn.Dropout2d(p=0.3)
        self.upconv3 = nn.ConvTranspose2d(features_list[1], features_list[0], kernel_size=2, stride=2)
        self.att3 = AttentionBlock(f_g=features_list[0], f_l=features_list[0], f_int=features_list[0]//2)
        self.dec3 = ResidualBlock(features_list[0] + features_list[0], features_list[0])
        self.dropout6 = nn.Dropout2d(p=0.2)
        self.final_conv = nn.Conv2d(features_list[0], out_channels, kernel_size=1)
        self.output_activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        enc1_out = self.enc1(x); pool1_out = self.pool1(enc1_out); pool1_out = self.dropout1(pool1_out)
        enc2_out = self.enc2(pool1_out); pool2_out = self.pool2(enc2_out); pool2_out = self.dropout2(pool2_out)
        enc3_out = self.enc3(pool2_out); pool3_out = self.pool3(enc3_out); pool3_out = self.dropout3(pool3_out)
        bottleneck_out = self.bottleneck(pool3_out); bottleneck_out = self.dropout_bottleneck(bottleneck_out)

        up1 = self.upconv1(bottleneck_out)
        if up1.shape[2] != enc3_out.shape[2] or up1.shape[3] != enc3_out.shape[3]:
            up1 = nn.functional.interpolate(up1, size=(enc3_out.shape[2], enc3_out.shape[3]), mode='nearest')
        att1_out = self.att1(g=up1, x=enc3_out)
        dec1_in = torch.cat([up1, att1_out], dim=1)
        dec1_out = self.dec1(dec1_in); dec1_out = self.dropout4(dec1_out)

        up2 = self.upconv2(dec1_out)
        if up2.shape[2] != enc2_out.shape[2] or up2.shape[3] != enc2_out.shape[3]:
            up2 = nn.functional.interpolate(up2, size=(enc2_out.shape[2], enc2_out.shape[3]), mode='nearest')
        att2_out = self.att2(g=up2, x=enc2_out)
        dec2_in = torch.cat([up2, att2_out], dim=1)
        dec2_out = self.dec2(dec2_in); dec2_out = self.dropout5(dec2_out)

        up3 = self.upconv3(dec2_out)
        if up3.shape[2] != enc1_out.shape[2] or up3.shape[3] != enc1_out.shape[3]:
            up3 = nn.functional.interpolate(up3, size=(enc1_out.shape[2], enc1_out.shape[3]), mode='nearest')
        att3_out = self.att3(g=up3, x=enc1_out)
        dec3_in = torch.cat([up3, att3_out], dim=1)
        dec3_out = self.dec3(dec3_in); dec3_out = self.dropout6(dec3_out)

        final_output = self.final_conv(dec3_out)
        output = self.output_activation(final_output)
        return output

# --- 5. Custom Loss Function (Unchanged) ---
class MaskedMSELoss(nn.Module):
    def __init__(self, signal_weight=10.0, background_weight=1.0, signal_threshold=1e-6):
        super(MaskedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.signal_weight = signal_weight
        self.background_weight = background_weight
        self.signal_threshold = signal_threshold

    def forward(self, inputs, targets, mask):
        element_wise_mse = self.mse_loss(inputs, targets)
        weighted_mse = (element_wise_mse * mask * self.signal_weight) + \
                       (element_wise_mse * (1.0 - mask) * self.background_weight)
        return torch.mean(weighted_mse)

# --- 6. Denoising Model Class (Unchanged) ---
class DenoisingModel:
    def __init__(self, model_params, training_params):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = DenoisingUNet(
            in_channels=model_params['in_channels'],
            out_channels=model_params['out_channels'],
            features_list=model_params['features_list']
        ).to(self.device)

        self.criterion = MaskedMSELoss(
            signal_weight=training_params['signal_weight'],
            background_weight=training_params['background_weight'],
            signal_threshold=training_params['signal_threshold']
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_params['learning_rate'],
            weight_decay=training_params['weight_decay']
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=training_params['scheduler_factor'],
            patience=training_params['scheduler_patience'],
        )

        #print("PyTorch U-Net Model:")
        #print(self.model)
        #print("-" * 30)

    def train_model(self, train_loader, val_loader, num_epochs, early_stopping_patience=25):
        print("[INFO] Starting model training...")
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_wts = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, targets, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]"):
                inputs, targets, masks = inputs.to(self.device), targets.to(self.device), masks.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)

            self.model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for inputs, targets, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VALID]"):
                    inputs, targets, masks = inputs.to(self.device), targets.to(self.device), masks.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets, masks)
                    val_running_loss += loss.item() * inputs.size(0)

            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            val_losses.append(val_epoch_loss)

            self.scheduler.step(val_epoch_loss)

            if val_epoch_loss < best_val_loss:
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

    def denoise_histogram(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            outputs_scaled = self.model(input_tensor.to(self.device)).cpu().numpy()

        denoised_outputs = np.zeros_like(outputs_scaled)
        for i in range(outputs_scaled.shape[0]):
            temp_output = outputs_scaled[i, 0, :, :] / amplitude_scaling_factor

            slice_sum = temp_output.sum()
            if slice_sum > 0:
                denoised_outputs[i, 0, :, :] = temp_output / slice_sum
            else:
                denoised_outputs[i, 0, :, :] = np.ones_like(temp_output) / (temp_output.shape[0] * temp_output.shape[1])
        return denoised_outputs


# --- 7. Plotting Functions (Refactored) ---
def plot_distributions_before(histograms_mmap, ct_vals, initial_e_vals, m_indices, e_indices, x_angles_np, y_final_energies_np, directory="./plots"):
    print(f'Plotting distributions for specified (M, E) pairs before denoising...')
    plot_counter = 0
    # Use np.min and np.max to be robust to tuple/array inputs
    extent = [np.min(x_angles_np), np.max(x_angles_np), np.min(y_final_energies_np), np.max(y_final_energies_np)]
    
    for m_idx_to_plot in m_indices:
        for e_idx_to_plot in e_indices:
            noisy_1 = histograms_mmap[0, m_idx_to_plot, e_idx_to_plot, :, :]
            noisy_2 = histograms_mmap[1, m_idx_to_plot, e_idx_to_plot, :, :]

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f'Distributions Before Denoising (M: {ct_vals[m_idx_to_plot]:.0f} CT, E: {initial_e_vals[e_idx_to_plot]:.0f} keV)', fontsize=16)

            im1 = axes[0].imshow(noisy_1.T, origin='lower', aspect='auto', cmap='viridis', extent=extent)
            axes[0].set_title('Noisy Input (Poisson Noise 1)')
            axes[0].set_xlabel('Transformed scattered Angle')
            axes[0].set_ylabel('Transformed Final Energy')
            plt.colorbar(im1, ax=axes[0], label='Probability')

            im2 = axes[1].imshow(noisy_2.T, origin='lower', aspect='auto', cmap='viridis', extent=extent)
            axes[1].set_title('Noisy Input (Poisson Noise 2)')
            axes[1].set_xlabel('Transformed scattered Angle')
            axes[1].set_ylabel('Transformed Final Energy')
            plt.colorbar(im2, ax=axes[1], label='Probability')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'{directory}/noisy_example_M{m_idx_to_plot}_E{e_idx_to_plot}.pdf', bbox_inches='tight')
            plt.close(fig)
            plot_counter += 1
            if plot_counter >= 1050:
                break

def plot_denoising_results(denoising_model, histograms_mmap, ct_vals, initial_e_vals, m_indices, e_indices, M, E, A, e, x_angles_np, y_final_energies_np, amplitude_scaling_factor, directory = "./plots"):
    print("Plotting denoising results for specified (M, E) pairs...")
    
    # Bug Fix: Ensure inputs are numpy arrays for min/max operations to prevent AttributeErrors
    x_angles_np = np.asarray(x_angles_np)
    y_final_energies_np = np.asarray(y_final_energies_np)

    plot_counter = 0
    # The fix here is to use the global np.min and np.max functions
    extent = [np.min(x_angles_np), np.max(x_angles_np), np.min(y_final_energies_np), np.max(y_final_energies_np)]
    
    # Pre-calculate coordinate grids and scaling once
    x_coords = torch.linspace(0, 1, A, dtype=torch.float32)
    y_coords = torch.linspace(0, 1, e, dtype=torch.float32)
    x_grid = x_coords.view(A, 1).expand(A, e)
    y_grid = y_coords.view(1, e).expand(A, e)
    ct_min, ct_max = np.min(ct_vals), np.max(ct_vals)
    e_min, e_max = np.min(initial_e_vals), np.max(initial_e_vals)

    for m_idx_to_plot in m_indices:
        for e_idx_to_plot in e_indices:
            noisy_input_np = histograms_mmap[0, m_idx_to_plot, e_idx_to_plot, :, :]
            noisy_target_np = histograms_mmap[1, m_idx_to_plot, e_idx_to_plot, :, :]

            noisy_input_img_scaled = torch.from_numpy(noisy_input_np).float() * amplitude_scaling_factor
            
            ct_scale = (ct_vals[m_idx_to_plot] - ct_min) / (ct_max - ct_min + 1e-12)
            e_scale = (initial_e_vals[e_idx_to_plot] - e_min) / (e_max - e_min + 1e-12)
            
            ct_channel = torch.as_tensor(ct_scale, dtype=torch.float32).expand(A, e)
            e_channel = torch.as_tensor(e_scale, dtype=torch.float32).expand(A, e)
            
            single_input_tensor = torch.stack(
                [noisy_input_img_scaled, ct_channel, e_channel, x_grid, y_grid], dim=0).unsqueeze(0)

            # Get noisy input data (channel 0)
            noisy_input_img = single_input_tensor[0, 0, :, :].numpy() / amplitude_scaling_factor
            
            # Denoise the single sample
            predicted_clean_img = denoising_model.denoise_histogram(single_input_tensor)[0, 0, :, :]
            
            true_clean_img = noisy_target_np
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Denoising Example (M: {ct_vals[m_idx_to_plot]:.0f} CT, E: {initial_e_vals[e_idx_to_plot]:.0f} keV)', fontsize=16)

            im1 = axes[0].imshow(noisy_input_img.T, origin='lower', aspect='auto', cmap='viridis', extent=extent)
            axes[0].set_title('Noisy Input (Poisson Noise 1)')
            axes[0].set_xlabel('Scattered Angle')
            axes[0].set_ylabel('Final Energy')
            plt.colorbar(im1, ax=axes[0], label='Probability')

            im2 = axes[1].imshow(predicted_clean_img.T, origin='lower', aspect='auto', cmap='viridis', extent=extent)
            axes[1].set_title('Predicted Denoised (Noise2Noise)')
            axes[1].set_xlabel('Scattered Angle')
            axes[1].set_ylabel('Final Energy')
            plt.colorbar(im2, ax=axes[1], label='Probability')

            im3 = axes[2].imshow(true_clean_img.T, origin='lower', aspect='auto', cmap='viridis', extent=extent)
            axes[2].set_title('True Clean')
            axes[2].set_xlabel('Scattered Angle')
            axes[2].set_ylabel('Final Energy')
            plt.colorbar(im3, ax=axes[2], label='Probability')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'{directory}/denoising_example_M{m_idx_to_plot}_E{e_idx_to_plot}.pdf', bbox_inches='tight')
            plt.close(fig)
            plot_counter += 1
            if plot_counter >= 1050:
                break


# --- 8. Function to Save Denoised Histograms (Unchanged logic, but uses new Dataset) ---
def save_denoised_histograms(denoising_model, histograms_mmap, npz_path, output_filename, original_shape, hu_values, initial_energies, method, extra=None, batch_size=30):
    """
    Denoises the full dataset by loading data on-the-fly and writes the
    denoised histograms and metadata to an NPZ file.
    """
    output_filename = f"{output_filename}_{method}.npz"
    print(f"\nDenoising full dataset and saving to {output_filename}…")

    M, E, A, e = original_shape
    
    full_dataset = OnTheFlyNPZDataset(npz_path, histograms_mmap, hu_values, initial_energies, M, E, A, e, 
                                      augment_data=False, amplitude_scaling_factor=amplitude_scaling_factor)

    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    
    denoised_outputs_list = []
    
    # We will get the raw model output and apply normalization/zero-check later
    denoising_model.model.eval()
    with torch.no_grad():
        for inputs, _, _ in tqdm(full_loader, desc="Denoising batches"):
            inputs = inputs.to(denoising_model.device)
            # Run the model to get the raw, un-normalized output
            raw_denoised = denoising_model.model(inputs)
            
            # Move both noisy input and raw output to CPU for processing
            raw_denoised_np = raw_denoised.cpu().numpy()
            noisy_inputs_np = inputs.cpu().numpy()

            # Create a new batch for corrected outputs
            corrected_batch_denoised = np.zeros_like(raw_denoised_np)

            # Loop through each sample in the batch to apply logic
            for i in range(raw_denoised_np.shape[0]):
                noisy_slice = noisy_inputs_np[i, 0, :, :]
                denoised_slice = raw_denoised_np[i, 0, :, :]

                # Check if the noisy input for this slice was all zeros
                # We use a small epsilon to account for potential floating point issues
                if np.sum(noisy_slice) < 1e-12:
                    # If the noisy input was zero, the output must be zero
                    corrected_batch_denoised[i, 0, :, :] = np.zeros_like(denoised_slice)
                else:
                    # Otherwise, normalize the denoised output
                    slice_sum = np.sum(denoised_slice)
                    if slice_sum > 1e-12:
                        corrected_batch_denoised[i, 0, :, :] = denoised_slice / slice_sum
                    else:
                        # Handle case where denoised output is also close to zero
                        # but noisy input was not. This is a fallback to a uniform
                        # distribution, but should be rare.
                        corrected_batch_denoised[i, 0, :, :] = np.ones_like(denoised_slice) / (denoised_slice.shape[0] * denoised_slice.shape[1])
            
            denoised_outputs_list.append(corrected_batch_denoised)

    all_denoised = np.concatenate(denoised_outputs_list, axis=0)
    # The output from the model is 4D (batch, channels, H, W). We reshape it to the 4D of the dataset (M, E, A, e).
    denoised_4d = all_denoised.reshape(original_shape)

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
    print(f"Denoised 4-D data (and method-specific variables) written to {output_filename}")


# --- Main Execution Block (Refactored) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a denoising CNN with real data from an NPZ file.")
    data_source_group = parser.add_mutually_exclusive_group(required=True)
    data_source_group.add_argument('--transformation', action='store_true', help="Use transformation method")
    data_source_group.add_argument('--normalization', action='store_true', help="Use normalization method")
    parser.add_argument("--npz", type=str, default=None, help="Optional path to a custom NPZ file.")
    args = parser.parse_args()
    
    # 1. Data Loading and Initial Setup
    method = 'transformation' if args.transformation else 'normalization'
    if args.npz:
        npz_path = args.npz
    else:
        npz_path = ('./DenoisingDataTransSheet.npz' if method == 'transformation' else './DenoisingDataNormSheet.npz')
    
    if not os.path.isfile(npz_path):
        print(f"[ERROR] NPZ file '{npz_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Load all metadata and a memory-mapped object for the histograms
    npz_path, histograms_mmap, num_materials, num_initial_energies, num_angles, num_final_energies, ct_values, initial_energies, method, extra = load_data_and_metadata_from_npz(method=method, npz_path=npz_path)

    original_4d_shape = (num_materials, num_initial_energies, num_angles, num_final_energies)
    print(f"[INFO] Original 4D shape: {original_4d_shape}")

    # 2. Create the efficient on-the-fly dataset
    full_dataset = OnTheFlyNPZDataset(
        npz_path=npz_path,
        histograms_mmap=histograms_mmap,
        ct_vals=ct_values,
        initial_e_vals=initial_energies,
        M=num_materials, E=num_initial_energies, A=num_angles, e=num_final_energies,
        augment_data=True,
        amplitude_scaling_factor=amplitude_scaling_factor
    )
    
    # 3. Define indices to plot for visualization
    if method == 'transformation':
        x_angles_range = np.linspace(0, 70, num_angles) 
        y_final_energies_range = np.linspace(-0.6, 0, num_final_energies) 
    else:
        x_angles_range = np.linspace(0, 1, num_angles)
        y_final_energies_range = np.linspace(0, 1, num_final_energies)
    
    print(f'[INFO] USing values for plotting x: ({x_angles_range.min()}, {x_angles_range.max()}) and y: ({y_final_energies_range.min()}, {y_final_energies_range.max()}) for visualization.')

    num_materials_to_plot = min(num_materials, 10)
    num_initial_energies_to_plot = min(num_initial_energies, 10)
    m_indices_to_plot_final = np.linspace(0, num_materials - 1, num_materials_to_plot, dtype=int)
    e_indices_to_plot_final = np.linspace(0, num_initial_energies - 1, num_initial_energies_to_plot, dtype=int)

    # Create the directory for saving plots
    directory = "./plots"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 4. Split dataset and create DataLoaders
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_dataset, val_dataset = random_split(
        full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    print(f"[INFO] Number of training samples: {len(train_dataset)}")
    print(f"[INFO] Number of validation samples: {len(val_dataset)}")
    
    # 5. Model Initialization and Training
    model_params = {
        "in_channels": 5,
        "out_channels": 1,
        "features_list": [16, 32, 64, 128]
    }
    
    print(f'[INFO] Model Parameters:')
    print(f'\t.In Channels: {model_params["in_channels"]}')
    print(f'\t.Out Channels: {model_params["out_channels"]}')
    print(f'\t.Features List: {model_params["features_list"]}')
    
    if method == 'transformation':
        signal_weight = 10.0
        background_weight = 10.0
    else:
        signal_weight = 5.0
        background_weight = 1.0
    training_params = {
        "signal_weight": signal_weight, "background_weight": background_weight, "signal_threshold": 1e-5 * amplitude_scaling_factor,
        "learning_rate": 0.00001, "weight_decay": 1e-5, "num_epochs": 200,
        "scheduler_factor": 0.1, "scheduler_patience": 5
    }
    
    print(f'[INFO] Training Parameters:')
    print(f'\t.Signal Weight: {training_params["signal_weight"]}')
    print(f'\t.Background Weight: {training_params["background_weight"]}')
    print(f'\t.Signal Threshold: {training_params["signal_threshold"]}')
    print(f'\t.Learning Rate: {training_params["learning_rate"]}')
    print(f'\t.Weight Decay: {training_params["weight_decay"]}')
    print(f'\t.Number of Epochs: {training_params["num_epochs"]}')
    print(f'\t.Scheduler Factor: {training_params["scheduler_factor"]}')
    print(f'\t.Scheduler Patience: {training_params["scheduler_patience"]}')

    denoising_model = DenoisingModel(model_params, training_params)
    print('[INFO] Number of parameters in the model:', sum(p.numel() for p in denoising_model.model.parameters() if p.requires_grad))

    train_losses, val_losses = denoising_model.train_model(
        train_loader, val_loader, training_params['num_epochs'], early_stopping_patience=5
    )
    # 6. Loss Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Model Loss (Weighted MSE)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('training_validation_loss.png', dpi=300)
    plt.close()
    print("[INFO] Training complete. Loss plots saved.")

    # 7. Visualize Denoising Results
    plot_denoising_results(
        denoising_model,
        histograms_mmap,
        ct_values,
        initial_energies,
        m_indices_to_plot_final,
        e_indices_to_plot_final,
        num_materials, num_initial_energies, num_angles, num_final_energies,
        x_angles_range,
        y_final_energies_range,
        amplitude_scaling_factor,
        directory=directory
    )
    print("[INFO] Denoising results plotted and saved.")
    
    # 8. Save Denoised Histograms
    save_denoised_histograms(
        denoising_model=denoising_model,
        histograms_mmap=histograms_mmap,
        npz_path=npz_path,
        output_filename="denoised_histograms",
        original_shape=original_4d_shape,
        hu_values=ct_values,
        initial_energies=initial_energies,
        method=method,
        extra=extra,
        batch_size=128
    )

    print("\n[INFO] Optimized U-Net based CNN for Noise2Noise denoising is complete.")
    
    # Print keys in the denoised histograms
    output_filename = f"denoised_histograms_{method}.npz"
    histogrmas = np.load(output_filename, allow_pickle=True)
    for key in histogrmas.keys():
        print(f"Key: {key}, Shape: {histogrmas[key].shape}")