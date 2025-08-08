import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
import copy
import argparse
import os
import optuna

# --- 1. Define Dummy Data Dimensions ---
num_materials = 50
num_initial_energies = 50
num_angles = 100
num_final_energies = 100

print(f"Dataset dimensions:")
print(f" Materials (M): {num_materials}")
print(f" Initial Energies (E): {num_initial_energies}")
print(f" Scattered Angles (a): {num_angles}")
print(f" Final Energies (e): {num_final_energies}")
print("-" * 30)

x_angles_range = np.linspace(0, 70, num_angles)
y_final_energies_range = np.linspace(-0.6, 0, num_final_energies)

amplitude_scaling_factor = 1000.0
print(f"Amplitude Scaling Factor: {amplitude_scaling_factor}")
print("-" * 30)

# --- 2. Data Generation Function ---
def generate_synthetic_noisy_data(M_dim, E_dim, A_Dim, e_dim, x_angles_vals, y_final_energies_vals, poisson_scaling_factor=1000):
    """
    Generates synthetic 4D noisy (noisy_h1, noisy_h2) and clean histogram data for testing.
    The clean data will have a more circular/Gaussian-like peak in (a,e) space that shifts
    based on M and E indices. Poisson noise is added only where there is signal.
    """
    print(f"Generating synthetic data with dimensions: M={M_dim}, E={E_dim}, A={A_Dim}, e={e_dim}")

    ct_values = np.linspace(-1000, 3000, M_dim)
    print(f"Dummy CT values for materials: {ct_values}")

    initial_energies = np.linspace(10, 100, E_dim)
    print(f"Dummy Initial Energy values: {initial_energies}")

    clean_target_data = torch.zeros((M_dim, E_dim, A_Dim, e_dim), dtype=torch.float32)

    a_grid = torch.tensor(x_angles_vals).float()
    e_grid = torch.tensor(y_final_energies_vals).float()

    a_mesh, e_mesh = torch.meshgrid(a_grid, e_grid, indexing='ij')
    pos = torch.empty(a_mesh.shape + (2,), dtype=torch.float32)
    pos[:, :, 0] = a_mesh
    pos[:, :, 1] = e_mesh

    angle_padding_abs = 0.1 * (x_angles_vals.max() - x_angles_vals.min())
    angle_mean_min = x_angles_vals.min() + angle_padding_abs
    angle_mean_max = x_angles_vals.max() - angle_padding_abs

    energy_padding_abs = 0.1 * (y_final_energies_vals.max() - y_final_energies_vals.min())
    final_energy_mean_min = y_final_energies_vals.min() + energy_padding_abs
    final_energy_mean_max = y_final_energies_vals.max() - energy_padding_abs

    base_cov_angle = 0.15
    angle_cov_variation = 0.05
    base_cov_final_energy = 10
    final_energy_cov_variation = 5

    print("\nGenerating clean 4D histogram data with M and E dependencies...")

    for m_idx in range(M_dim):
        for e_idx in range(E_dim):
            norm_m = m_idx / (M_dim - 1) if M_dim > 1 else 0.5
            norm_e = e_idx / (E_dim - 1) if E_dim > 1 else 0.5

            mean_angle = angle_mean_min + (angle_mean_max - angle_mean_min) * norm_e
            mean_final_energy = final_energy_mean_min + (final_energy_mean_max - final_energy_mean_min) * norm_m

            cov_angle = base_cov_angle - angle_cov_variation * norm_e
            cov_final_energy = base_cov_final_energy - final_energy_cov_variation * norm_m

            cov_angle = max(0.01, cov_angle)
            cov_final_energy = max(1.0, cov_final_energy)

            rv = multivariate_normal(mean=[mean_angle.item(), mean_final_energy.item()],
                                     cov=[[cov_angle**2, 0], [0, cov_final_energy**2]])

            pdf_values_clean_np = rv.pdf(pos.numpy())
            pdf_values_clean = torch.from_numpy(pdf_values_clean_np).float()

            slice_sum = pdf_values_clean.sum()
            if slice_sum > 0:
                current_slice = pdf_values_clean / slice_sum
            else:
                current_slice = torch.ones((A_Dim, e_dim), dtype=torch.float32) / (A_Dim * e_dim)

            clean_target_data[m_idx, e_idx, :, :] = current_slice

    non_zero_mask = (clean_target_data > 1e-6).float()

    clean_scaled_for_poisson = clean_target_data * poisson_scaling_factor

    poisson_noise1 = torch.poisson(torch.round(clean_scaled_for_poisson))
    poisson_noise2 = torch.poisson(torch.round(clean_scaled_for_poisson * 1.1))

    noisy_h1 = (poisson_noise1 / poisson_scaling_factor) * non_zero_mask
    noisy_h2_input_to_model = (poisson_noise2 / poisson_scaling_factor) * non_zero_mask

    noisy_h1 = torch.clamp(noisy_h1, min=0.0)
    noisy_h2_input_to_model = torch.clamp(noisy_h2_input_to_model, min=0.0)

    for m_idx in range(M_dim):
        for e_idx in range(E_dim):
            slice_sum_h1 = noisy_h1[m_idx, e_idx, :, :].sum()
            if slice_sum_h1 > 0:
                noisy_h1[m_idx, e_idx, :, :] /= slice_sum_h1
            else:
                noisy_h1[m_idx, e_idx, :, :] = torch.ones((A_Dim, e_dim), dtype=torch.float32) / (A_Dim * e_dim)

            slice_sum_h2 = noisy_h2_input_to_model[m_idx, e_idx, :, :].sum()
            if slice_sum_h2 > 0:
                noisy_h2_input_to_model[m_idx, e_idx, :, :] /= slice_sum_h2
            else:
                noisy_h2_input_to_model[m_idx, e_idx, :, :] = torch.ones((A_Dim, e_dim), dtype=torch.float32) / (A_Dim * e_dim)

    mask_volume_4d = (clean_target_data > 1e-6).bool()

    print(f"Synthetic Noisy Volume 1 (4D) Shape: {noisy_h1.shape}")
    print(f"Synthetic Noisy Volume 2 (4D) Shape: {noisy_h2_input_to_model.shape}")
    print(f"Synthetic Clean Target Volume (4D) Shape: {clean_target_data.shape}")
    print(f"Synthetic Mask Volume (4D) Shape: {mask_volume_4d.shape}")

    return noisy_h1, noisy_h2_input_to_model, clean_target_data, mask_volume_4d, M_dim, E_dim, A_Dim, e_dim, ct_values, initial_energies

# --- 2a. Data Loading Function ---
def load_data_from_npz(npz_path="./DenoisingDataTransSheet50.npz"):
    """
    Loads pre-generated noisy data from a .npz file and prepares it for the pipeline.
    The file is expected to contain a 5D array 'histograms' of shape [2, M, E, a, e].
    """
    print(f"Loading data from: {npz_path}")
    try:
        loaded_data = np.load(npz_path)
        histograms_np = loaded_data['histograms'].astype(np.float32)
        ct_values = loaded_data['HU'].astype(np.float32)
        initial_energies = loaded_data['energies'].astype(np.float32)

    except FileNotFoundError:
        raise FileNotFoundError(f"The file {npz_path} was not found.")
    except KeyError:
        raise KeyError("The .npz file must contain 'histograms', 'HU', 'energies'.")

    if histograms_np.ndim != 5 or histograms_np.shape[0] != 2:
        raise ValueError(
            "Expected 'histograms' data to be 5D with shape [2, M, E, a, e]. "
            f"Got shape: {histograms_np.shape}"
        )

    # Separate the two noisy samples
    noisy_h1 = torch.from_numpy(histograms_np[0]) # Input to model
    noisy_h2 = torch.from_numpy(histograms_np[1]) # Target for training

    # Extract dimensions
    M_dim, E_dim, A_Dim, e_dim = noisy_h1.shape

    clean_target_data = noisy_h2.clone() # Using noisy_h2 as a proxy for plotting

    # The mask should identify where there's actual signal in the *clean* data, but since we don't have it,
    # we approximate it by looking at where there's signal in either noisy version.
    mask_volume_4d = (noisy_h1 + noisy_h2 > 1e-12).bool()

    print(f"Loaded Noisy Volume 1 (4D) Shape: {noisy_h1.shape}")
    print(f"Loaded Noisy Volume 2 (4D) Shape: {noisy_h2.shape}")
    print(f"Proxy Clean Target Volume (4D) Shape: {clean_target_data.shape}")
    print(f"Generated Mask Volume (4D) Shape: {mask_volume_4d.shape}")
    print("-" * 30)

    return (noisy_h1, noisy_h2, clean_target_data, mask_volume_4d,
            M_dim, E_dim, A_Dim, e_dim, ct_values, initial_energies)


# --- 3. Data Transformation Function ---
def create_cnn_inputs_pytorch(noisy_input_data, noisy_target_data, clean_mask_data, ct_vals, initial_e_vals, num_a, num_e, augment_data=False):
    X_cnn_images = []
    y_target_images = []
    mask_images = []

    min_e, max_e = np.min(initial_e_vals), np.max(initial_e_vals)
    min_ct, max_ct = np.min(ct_vals), np.max(ct_vals)

    scaled_initial_e_vals = (initial_e_vals - min_e) / (max_e - min_e) if (max_e - min_e) > 0 else np.zeros_like(initial_e_vals)
    scaled_ct_vals = (ct_vals - min_ct) / (max_ct - min_ct) if (max_ct - min_ct) > 0 else np.zeros_like(ct_vals)

    x_coords = torch.linspace(0, 1, num_a).float()
    y_coords = torch.linspace(0, 1, num_e).float()

    x_coord_grid = x_coords.view(num_a, 1).expand(num_a, num_e)
    y_coord_grid = y_coords.view(1, num_e).expand(num_a, num_e)

    for m_idx in range(noisy_input_data.shape[0]):
        for e_idx in range(noisy_input_data.shape[1]):
            hist_channel = noisy_input_data[m_idx, e_idx, :, :]
            energy_channel = torch.full((num_a, num_e), scaled_initial_e_vals[e_idx], dtype=torch.float32)
            ct_channel = torch.full((num_a, num_e), scaled_ct_vals[m_idx], dtype=torch.float32)
            angle_coord_channel = x_coord_grid
            final_energy_coord_channel = y_coord_grid

            combined_input_image = torch.stack([
                hist_channel,
                energy_channel,
                ct_channel,
                angle_coord_channel,
                final_energy_coord_channel
            ], dim=0)

            hist_target = noisy_target_data[m_idx, e_idx, :, :].unsqueeze(0)
            hist_mask = clean_mask_data[m_idx, e_idx, :, :].unsqueeze(0).float()

            if augment_data:
                # Apply augmentations to all relevant channels
                all_channels_to_augment = torch.cat([combined_input_image, hist_target, hist_mask], dim=0)

                # The apply_augmentations function now handles multiple channels
                augmented_channels = all_channels_to_augment
                # Randomly flip horizontally
                if np.random.rand() > 0.5:
                    augmented_channels = torch.flip(augmented_channels, dims=[1])
                # Randomly flip vertically
                if np.random.rand() > 0.5:
                    augmented_channels = torch.flip(augmented_channels, dims=[2])

                combined_input_image = augmented_channels[:5]
                hist_target = augmented_channels[5].unsqueeze(0)
                hist_mask = augmented_channels[6].unsqueeze(0)

            X_cnn_images.append(combined_input_image)
            y_target_images.append(hist_target)
            mask_images.append(hist_mask)

    return torch.stack(X_cnn_images, dim=0), torch.stack(y_target_images, dim=0), torch.stack(mask_images, dim=0)


# --- 4. Robust U-Net Architecture with Residual, Attention Blocks, and Dropout ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)
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
        out = self.dropout(out) # Dropout applied after first ReLU
        out = self.conv2(out)
        out = self.in2(out)
        out += identity
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(f_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DenoisingUNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, features_list=None, dropout_p=0.2):
        super(DenoisingUNet, self).__init__()

        if features_list is None:
            features_list = [32, 64, 128, 256]
        
        # Adjust dropout probabilities based on a single parameter for consistency
        # Using a fixed progression for dropout across layers for simplicity and control
        dropout_p_enc = [dropout_p * 0.5, dropout_p * 0.7, dropout_p * 0.9] # For ResidualBlocks
        dropout_p_pool = [dropout_p * 0.6, dropout_p * 0.8, dropout_p * 1.0] # For Dropout2d after pooling
        dropout_p_dec = [dropout_p * 0.9, dropout_p * 0.7, dropout_p * 0.5] # For ResidualBlocks and Dropout2d in decoder

        # Encoder Path with Residual Blocks and Dropout
        self.enc1 = ResidualBlock(in_channels, features_list[0], dropout_p=dropout_p_enc[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=dropout_p_pool[0])

        self.enc2 = ResidualBlock(features_list[0], features_list[1], dropout_p=dropout_p_enc[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=dropout_p_pool[1])

        self.enc3 = ResidualBlock(features_list[1], features_list[2], dropout_p=dropout_p_enc[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=dropout_p_pool[2])

        # Bottleneck with Residual Block and Dropout
        self.bottleneck = ResidualBlock(features_list[2], features_list[3], dropout_p=dropout_p) # Can be highest dropout
        self.dropout_bottleneck = nn.Dropout2d(p=dropout_p)

        # Decoder Path with Attention, Residual Blocks, and Dropout
        self.upconv1 = nn.ConvTranspose2d(features_list[3], features_list[2], kernel_size=2, stride=2)
        self.att1 = AttentionBlock(f_g=features_list[2], f_l=features_list[2], f_int=features_list[2]//2)
        self.dec1 = ResidualBlock(features_list[2] + features_list[2], features_list[2], dropout_p=dropout_p_dec[0])
        self.dropout4 = nn.Dropout2d(p=dropout_p_dec[0])

        self.upconv2 = nn.ConvTranspose2d(features_list[2], features_list[1], kernel_size=2, stride=2)
        self.att2 = AttentionBlock(f_g=features_list[1], f_l=features_list[1], f_int=features_list[1]//2)
        self.dec2 = ResidualBlock(features_list[1] + features_list[1], features_list[1], dropout_p=dropout_p_dec[1])
        self.dropout5 = nn.Dropout2d(p=dropout_p_dec[1])

        self.upconv3 = nn.ConvTranspose2d(features_list[1], features_list[0], kernel_size=2, stride=2)
        self.att3 = AttentionBlock(f_g=features_list[0], f_l=features_list[0], f_int=features_list[0]//2)
        self.dec3 = ResidualBlock(features_list[0] + features_list[0], features_list[0], dropout_p=dropout_p_dec[2])
        self.dropout6 = nn.Dropout2d(p=dropout_p_dec[2])

        self.final_conv = nn.Conv2d(features_list[0], out_channels, kernel_size=1)
        self.output_activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        pool1_out = self.pool1(enc1_out)
        pool1_out = self.dropout1(pool1_out)

        enc2_out = self.enc2(pool1_out)
        pool2_out = self.pool2(enc2_out)
        pool2_out = self.dropout2(pool2_out)

        enc3_out = self.enc3(pool2_out)
        pool3_out = self.pool3(enc3_out)
        pool3_out = self.dropout3(pool3_out)

        # Bottleneck
        bottleneck_out = self.bottleneck(pool3_out)
        bottleneck_out = self.dropout_bottleneck(bottleneck_out)

        # Decoder with Attention & Skip Connections
        up1 = self.upconv1(bottleneck_out)
        if up1.shape[2] != enc3_out.shape[2] or up1.shape[3] != enc3_out.shape[3]:
            up1 = nn.functional.interpolate(up1, size=(enc3_out.shape[2], enc3_out.shape[3]), mode='nearest')
        att1_out = self.att1(g=up1, x=enc3_out)
        dec1_in = torch.cat([up1, att1_out], dim=1)
        dec1_out = self.dec1(dec1_in)
        dec1_out = self.dropout4(dec1_out)

        up2 = self.upconv2(dec1_out)
        if up2.shape[2] != enc2_out.shape[2] or up2.shape[3] != enc2_out.shape[3]:
            up2 = nn.functional.interpolate(up2, size=(enc2_out.shape[2], enc2_out.shape[3]), mode='nearest')
        att2_out = self.att2(g=up2, x=enc2_out)
        dec2_in = torch.cat([up2, att2_out], dim=1)
        dec2_out = self.dec2(dec2_in)
        dec2_out = self.dropout5(dec2_out)

        up3 = self.upconv3(dec2_out)
        if up3.shape[2] != enc1_out.shape[2] or up3.shape[3] != enc1_out.shape[3]:
            up3 = nn.functional.interpolate(up3, size=(enc1_out.shape[2], enc1_out.shape[3]), mode='nearest')
        att3_out = self.att3(g=up3, x=enc1_out)
        dec3_in = torch.cat([up3, att3_out], dim=1)
        dec3_out = self.dec3(dec3_in)
        dec3_out = self.dropout6(dec3_out)

        final_output = self.final_conv(dec3_out)
        output = self.output_activation(final_output)

        return output

# --- 5. Custom Loss Function ---
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

# --- 6. Denoising Model Class (Encapsulates U-Net, Loss, Optimizer, Training) ---
class DenoisingModel:
    def __init__(self, model_params, training_params):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = DenoisingUNet(
            in_channels=model_params['in_channels'],
            out_channels=model_params['out_channels'],
            features_list=model_params['features_list'],
            dropout_p=model_params['dropout_p'] # Pass dropout_p to UNet
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

        print("PyTorch U-Net Model:")
        print(self.model)
        print("-" * 30)

    def train_model(self, train_loader, val_loader, num_epochs, early_stopping_patience=15, trial=None):
        print("Starting model training...")
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_wts = None
        epochs_without_improvement = 0  # Initialize the counter outside the loop

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, targets, masks in train_loader:
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
                for inputs, targets, masks in val_loader:
                    inputs, targets, masks = inputs.to(self.device), targets.to(self.device), masks.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets, masks)
                    val_running_loss += loss.item() * inputs.size(0)

            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            val_losses.append(val_epoch_loss)

            self.scheduler.step(val_epoch_loss)

            # Check for improvement and update counter
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0  # Reset the counter
                print(f'Epoch {epoch+1}/{num_epochs}, NEW BEST VAL LOSS: {val_epoch_loss:.4f}. Model weights saved.')
                # Save the best model weights to disk
                torch.save(best_model_wts, 'best_model_weights.pth')
            else:
                epochs_without_improvement += 1  # Increment the counter
                print(f'Epoch {epoch+1}/{num_epochs}, No Improvement. Val Loss: {val_epoch_loss:.4f}')

            # Report intermediate value to Optuna if a trial is active
            if trial:
                trial.report(val_epoch_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # Check for early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1} as validation loss has not improved for {early_stopping_patience} consecutive epochs.')
                break

            print(f'Current Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')

        print("Model training finished.")
        print("-" * 30)

        # Load the best weights before returning
        if best_model_wts:
            self.model.load_state_dict(best_model_wts)
        else:
            print("Warning: No model weights were saved as validation loss never improved.")

        # Return all three values for flexibility (especially for Optuna)
        return train_losses, val_losses, best_val_loss

    def denoise_histogram(self, input_tensor):
        """
        Denoises a single histogram or a batch of histograms.
        Input tensor should be in the format [N, C, H, W] where C=5.
        Returns denoised output in [N, 1, H, W] format, scaled back to original probability range.
        """
        self.model.eval()
        with torch.no_grad():
            outputs_scaled = self.model(input_tensor.to(self.device)).cpu().numpy()

        denoised_outputs = np.zeros_like(outputs_scaled)
        for i in range(outputs_scaled.shape[0]):
            temp_output = outputs_scaled[i, 0, :, :] # No division by amplitude_scaling_factor here, as it's handled in the plotting/saving
            slice_sum = temp_output.sum()
            if slice_sum > 0:
                denoised_outputs[i, 0, :, :] = temp_output / slice_sum
            else:
                denoised_outputs[i, 0, :, :] = np.ones_like(temp_output) / (temp_output.shape[0] * temp_output.shape[1])
        return denoised_outputs


# --- 7. Data Loader Preparation Function ---
def prepare_dataloaders(X_data, y_data_noisy, mask_data, y_data_clean, batch_size, random_state=42):
    # Ensure all data is on CPU before splitting to avoid CUDA memory issues with sklearn
    X_data_cpu = X_data.cpu()
    y_data_noisy_cpu = y_data_noisy.cpu()
    mask_data_cpu = mask_data.cpu()
    y_data_clean_cpu = y_data_clean.cpu()

    X_train, X_val, \
    y_train_noisy, y_val_noisy, \
    mask_train, mask_val, \
    y_train_clean, y_val_clean = train_test_split(
        X_data_cpu, y_data_noisy_cpu, mask_data_cpu, y_data_clean_cpu,
        test_size=0.2, random_state=random_state
    )

    train_dataset = TensorDataset(X_train, y_train_noisy, mask_train)
    val_dataset = TensorDataset(X_val, y_val_noisy, mask_val)

    # Use pin_memory=True and num_workers for faster data loading on GPU
    # Check if CUDA is available before setting pin_memory and num_workers
    num_workers = 4 if torch.cuda.is_available() else 0 # Set num_workers to 0 for CPU
    pin_memory = torch.cuda.is_available() # Only pin memory if CUDA is available

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

    return train_loader, val_loader, X_val, y_val_clean

# --- 8. Plotting Function ---
def plot_distributions_before(noisy_input_data, noisy_target_data, m_indices, e_indices, ct_vals, initial_e_vals, x_angles_np, y_final_energies_np, directory="./plots"):
    print(f'Plotting distributions for specified (M, E) pairs before denoising...')
    plot_counter = 0
    extent = [x_angles_np.min(), x_angles_np.max(), y_final_energies_np.min(), y_final_energies_np.max()]
    for m_idx_to_plot in m_indices:
        for e_idx_to_plot in e_indices:
            noisy_1 = noisy_input_data[m_idx_to_plot, e_idx_to_plot, :, :].numpy()
            noisy_2 = noisy_target_data[m_idx_to_plot, e_idx_to_plot, :, :].numpy()

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
            plt.close(fig) # Close the figure to free up memory

            plot_counter += 1
            if plot_counter >= 1050:
                break

def plot_denoising_results(denoising_model, X_full_data, clean_full_data, m_indices, e_indices, ct_vals, initial_e_vals, x_angles_np, y_final_energies_np, amplitude_scaling_factor, directory = "./plots"):
    print("Plotting denoising results for specified (M, E) pairs...")
    plot_counter = 0
    extent = [x_angles_np.min(), x_angles_np.max(), y_final_energies_np.min(), y_final_energies_np.max()]
    for m_idx_to_plot in m_indices:
        for e_idx_to_plot in e_indices:
            flat_idx = m_idx_to_plot * num_initial_energies + e_idx_to_plot
            if flat_idx >= X_full_data.shape[0]:
                continue
                
            noisy_input_img_scaled = X_full_data[flat_idx, 0, :, :].numpy()
            
            single_input_tensor = X_full_data[flat_idx:flat_idx+1, :, :, :]

            # The denoise_histogram function already handles normalization,
            # so we just need to scale the input for plotting if it was scaled up initially
            predicted_clean_img = denoising_model.denoise_histogram(single_input_tensor)[0, 0, :, :]

            true_clean_img_scaled = clean_full_data[m_idx_to_plot, e_idx_to_plot, :, :].numpy()
            true_clean_img = true_clean_img_scaled # Assuming clean_full_data is already normalized or will be scaled down for plotting

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Denoising Example (M: {ct_vals[m_idx_to_plot]:.0f} CT, E: {initial_e_vals[e_idx_to_plot]:.0f} keV)', fontsize=16)

            im1 = axes[0].imshow(noisy_input_img_scaled.T / amplitude_scaling_factor, origin='lower', aspect='auto', cmap='viridis', extent=extent) # Scale down for display
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

# --- 9. Function to Save Denoised Histograms ---
def save_denoised_histograms(denoising_model, X_full_data, output_filename, original_shape, batch_size=30):
    """
    Denoises the entire dataset and saves the output to an NPZ file.

    Args:
        denoising_model (DenoisingModel): The trained denoising model.
        X_full_data (torch.Tensor): The full input data for the model.
        output_filename (str): The name of the NPZ file to save.
        original_shape (tuple): The original 4D shape (M, E, A, e) to reshape the output to.
        batch_size (int): Batch size for denoising.
    """
    print(f"\nDenoising full dataset and saving to {output_filename}...")
    
    # Create a DataLoader for the full dataset
    full_dataset = TensorDataset(X_full_data)
    # Use pin_memory and num_workers based on CUDA availability for the saving process too
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    
    denoised_outputs_list = []
    
    denoising_model.model.eval()
    with torch.no_grad():
        for inputs in full_loader:
            # inputs[0] because DataLoader returns a tuple even for a single tensor
            batch_denoised = denoising_model.denoise_histogram(inputs[0])
            denoised_outputs_list.append(batch_denoised)

    # Concatenate all denoised batches
    all_denoised = np.concatenate(denoised_outputs_list, axis=0)

    # Reshape the data to the original 4D structure (M, E, A, e)
    # The output from denoise_histogram is [N, 1, H, W], so we need to flatten the channel dimension
    denoised_4d = all_denoised.reshape(original_shape)

    # Save the data to an NPZ file with the key 'histograms'
    np.savez(output_filename, histograms=denoised_4d)
    
    print(f"Denoised 4D data saved successfully to {output_filename}")


def objective(trial, X_data, y_data_noisy, mask_data, y_data_clean, ct_values, initial_energies, num_angles, num_final_energies):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_epochs = trial.suggest_int("num_epochs", 50, 150)
    scheduler_patience = trial.suggest_int("scheduler_patience", 3, 10)
    features_base = trial.suggest_int("features_base", 8, 32, step=8)
    features_list = [features_base, features_base * 2, features_base * 4, features_base * 8]
    dropout_p = trial.suggest_float("dropout_p", 0.1, 0.5)
    
    # Prepare dataloaders with suggested batch size
    train_loader, val_loader, _, _ = prepare_dataloaders(
        X_data, y_data_noisy, mask_data, y_data_clean, batch_size
    )

    model_params = {
        "in_channels": 5,
        "out_channels": 1,
        "features_list": features_list,
        "dropout_p": dropout_p
    }

    training_params = {
        "signal_weight": 10.0,
        "background_weight": 1.0,
        "signal_threshold": 1e-5 * amplitude_scaling_factor,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "scheduler_factor": 0.1,
        "scheduler_patience": scheduler_patience
    }

    denoising_model = DenoisingModel(model_params, training_params)
    
    # Train the model and get the best validation loss, passing the trial object for pruning
    _, _, best_val_loss = denoising_model.train_model(train_loader, val_loader, num_epochs, early_stopping_patience=15, trial=trial)
    
    return best_val_loss

# --- Main Execution Block with Optuna Integration ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a denoising CNN with either dummy synthetic data or real data from an NPZ file.")
    data_source_group = parser.add_mutually_exclusive_group(required=True)
    data_source_group.add_argument("--dummy", action="store_true",
                                   help="Use dummy synthetic data for training.")
    data_source_group.add_argument("--real", dest="npz_path", metavar="NPZ_PATH",
                                   nargs='?', const="./DenoisingDataTransSheet50.npz",
                                   help="Use real data from an NPZ file. Optionally specify the file path. "
                                        "Defaults to './DenoisingDataTransSheet50.npz'.")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization with Optuna.")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials.")
    args = parser.parse_args()

    # --- Data Loading and Preprocessing ---
    noisy1_histograms = None
    noisy2_target = None
    clean_target_data_proxy = None
    mask_volume_4d = None
    num_materials = None
    num_initial_energies = None
    num_angles = None
    num_final_energies = None
    ct_values = None
    initial_energies = None

    if args.dummy:
        print("Using dummy synthetic data.")
        (noisy1_histograms, noisy2_target, clean_target_data_proxy, mask_volume_4d,
         num_materials, num_initial_energies, num_angles, num_final_energies,
         ct_values, initial_energies) = generate_synthetic_noisy_data(
             num_materials, num_initial_energies, num_angles, num_final_energies,
             x_angles_range, y_final_energies_range,
             poisson_scaling_factor=100000
         )
    elif args.npz_path:
        print(f"Using real data from NPZ file: {args.npz_path}")
        (noisy1_histograms, noisy2_target, clean_target_data_proxy, mask_volume_4d,
         num_materials, num_initial_energies, num_angles, num_final_energies,
         ct_values, initial_energies) = load_data_from_npz(args.npz_path)

    # Store the original 4D shape for reshaping the final output
    original_4d_shape = noisy1_histograms.shape

    # Scale the data for model input
    noisy1_histograms = noisy1_histograms * amplitude_scaling_factor
    noisy2_target = noisy2_target * amplitude_scaling_factor
    clean_target_data_proxy = clean_target_data_proxy * amplitude_scaling_factor # Keep scaled for plotting consistent with model output

    X_data_torch, y_data_torch_noisy_target, mask_data_torch = create_cnn_inputs_pytorch(
        noisy1_histograms, noisy2_target, mask_volume_4d,
        ct_values, initial_energies, num_angles, num_final_energies, augment_data=False
    )
    y_data_torch_clean_reshaped = clean_target_data_proxy.reshape(-1, 1, num_angles, num_final_energies)

    print(f"Input data shape: {X_data_torch.shape}, Target shape: {y_data_torch_noisy_target.shape}")
    print("-" * 30)

    # --- Hyperparameter Optimization or Default Run ---
    if args.optimize:
        print("Starting hyperparameter optimization with Optuna...")
        # Use a SQLite database to store study results for persistence and visualization
        study = optuna.create_study(direction="minimize", study_name="denoising_unet_optimization", storage="sqlite:///db.sqlite3", load_if_exists=True)
        study.optimize(lambda trial: objective(trial, X_data_torch, y_data_torch_noisy_target, mask_data_torch, y_data_torch_clean_reshaped, ct_values, initial_energies, num_angles, num_final_energies), n_trials=args.n_trials)

        print("\n--- Hyperparameter Optimization Finished ---")
        print(f"Best trial:")
        print(f" Value: {study.best_value}")
        print(" Params: ")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Use best hyperparameters for final model training
        best_params = study.best_params
        model_params = {
            "in_channels": 5,
            "out_channels": 1,
            "features_list": [best_params['features_base'], best_params['features_base'] * 2, best_params['features_base'] * 4, best_params['features_base'] * 8],
            "dropout_p": best_params['dropout_p']
        }
        training_params = {
            "signal_weight": 10.0,
            "background_weight": 1.0, # Keep background_weight at 1.0 for Optuna optimization
            "signal_threshold": 1e-5 * amplitude_scaling_factor,
            "learning_rate": best_params['learning_rate'],
            "weight_decay": best_params['weight_decay'],
            "num_epochs": best_params['num_epochs'],
            "scheduler_factor": 0.1,
            "scheduler_patience": best_params['scheduler_patience']
        }
        batch_size = best_params['batch_size']
        early_stopping_patience = 15 # Fixed patience for the final training run

    else:
        # Fallback to manual hyperparameters if not optimizing
        print("Running with default hyperparameters.")
        model_params = {
            "in_channels": 5,
            "out_channels": 1,
            "features_list": [8, 16, 32, 64],
            "dropout_p": 0.2
        }
        training_params = {
            "signal_weight": 10.0,
            "background_weight": 1.0,
            "signal_threshold": 1e-5 * amplitude_scaling_factor,
            "learning_rate": 0.00001,
            "weight_decay": 1e-5,
            "num_epochs": 100,
            "scheduler_factor": 0.1,
            "scheduler_patience": 5
        }
        batch_size = 30
        early_stopping_patience = 15 # Fixed patience for the final training run

    # Common training and plotting block for both optimization and manual runs
    directory = "./plots"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Prepare dataloaders for the final training run (or default run)
    train_loader, val_loader, X_val_for_plotting, y_val_clean_for_plotting = prepare_dataloaders(
        X_data_torch, y_data_torch_noisy_target, mask_data_torch, y_data_torch_clean_reshaped, batch_size
    )

    denoising_model = DenoisingModel(model_params, training_params)
    print('Number of parameters in the model:', sum(p.numel() for p in denoising_model.model.parameters() if p.requires_grad))

    # Train the final model with early stopping
    train_losses, val_losses, _ = denoising_model.train_model(
        train_loader, val_loader, training_params['num_epochs'], early_stopping_patience=early_stopping_patience
    )
    
    # Plot the loss curves for the final model
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
    print("Training complete. Loss plots saved.")

    # 7. Visualize Denoising Results for the final model
    num_materials_to_plot = min(num_materials, 3)
    num_initial_energies_to_plot = min(num_initial_energies, 3)
    m_indices_to_plot_final = np.linspace(0, num_materials - 1, num_materials_to_plot, dtype=int)
    e_indices_to_plot_final = np.linspace(0, num_initial_energies - 1, num_initial_energies_to_plot, dtype=int)

    plot_denoising_results(
        denoising_model,
        X_data_torch,
        clean_target_data_proxy, # This is the scaled clean data
        m_indices_to_plot_final,
        e_indices_to_plot_final,
        ct_values,
        initial_energies,
        x_angles_range,
        y_final_energies_range,
        amplitude_scaling_factor,
        directory=directory
    )
    print("Denoising results plotted and saved.")

    # 8. Save the full denoised model to an NPZ file
    # Note: The `denoise_histogram` function returns normalized probabilities,
    # so no need to divide by amplitude_scaling_factor here before saving.
    save_denoised_histograms(
        denoising_model=denoising_model,
        X_full_data=X_data_torch,
        output_filename="denoised_histograms.npz",
        original_shape=original_4d_shape
    )

    print("\nRefactored U-Net based CNN for Noise2Noise denoising is complete.")