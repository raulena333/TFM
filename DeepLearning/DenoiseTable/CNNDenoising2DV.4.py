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
import sys
from tqdm import tqdm  

# --- 1. Define Dummy Data Dimensions ---
num_materials = 50
num_initial_energies = 50
num_angles = 100
num_final_energies = 100

x_angles_range = np.linspace(0, 70, num_angles)
y_final_energies_range = np.linspace(-0.6, 0, num_final_energies)

amplitude_scaling_factor = 1000.0

# --- 2. Data Generation Function ---
def generate_synthetic_noisy_data(M_dim, E_dim, A_Dim, e_dim, x_angles_vals, y_final_energies_vals, poisson_scaling_factor=1000):
    print(f"Generating synthetic data with dimensions: M={M_dim}, E={E_dim}, A={A_Dim}, e={e_dim}")

    ct_values = np.linspace(-1000, 3000, M_dim)
    initial_energies = np.linspace(10, 100, E_dim)
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

    return noisy_h1, noisy_h2_input_to_model, clean_target_data, mask_volume_4d, M_dim, E_dim, A_Dim, e_dim, ct_values, initial_energies

# --- 2a. Data Loading Function ---
def load_data_from_npz(method, npz_path):
    
    print(f"[INFO] Loading data from: {npz_path}")
    
    method = method.lower()
    if method not in {"transformation", "normalization"}:
        raise ValueError(
            f"Unknown method '{method}'.  Expected 'transformation' "
            f"or 'normalization'."
        )
        
    try:
        npz = np.load(npz_path, allow_pickle=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"The file {npz_path} was not found.") from exc
    except Exception as exc:
        raise RuntimeError(f"Unable to load {npz_path}") from exc
    
    try:
        histograms_np = npz['histograms'].astype(np.float32)
    except KeyError as exc:
        raise KeyError("'histograms' key is missing from the .npz file") from exc

    try:
        ct_values = npz['HU'].astype(np.float32)
        initial_energies = npz['energies'].astype(np.float32)
    except KeyError as exc:
        raise KeyError("The .npz file must contain 'HU' and 'energies' keys") from exc
    
    if histograms_np.ndim != 5 or histograms_np.shape[0] != 2:
        raise ValueError("Expected 'histograms' data to be 5D with shape [2, M, E, a, e].")

    noisy_h1 = torch.from_numpy(histograms_np[0])
    noisy_h2 = torch.from_numpy(histograms_np[1])

    M_dim, E_dim, A_Dim, e_dim = noisy_h1.shape
    clean_target_data = noisy_h2.clone()
    mask_volume_4d = (noisy_h1 > 1e-12).bool()
    
    base_keys = {"histograms", "HU", "energies"}
    extra = {k: npz[k] for k in npz.files if k not in base_keys}
    
    required_common = {"rho"}
    missing = required_common - set(extra.keys())
    if missing:
        raise KeyError(
            f"For method '{method}' the following required key(s) are "
            f"missing in the .npz file: {', '.join(sorted(missing))}"
        )

    if method == "normalization":
        # 5 additional keys that are specific to the normalisation step
        required_norm = {"thetaMax", "thetaMin", "energyMin", "energyMax"}
        missing = required_norm - set(extra.keys())
        if missing:
            raise KeyError(
                f"For method 'normalization' the following required key(s) are "
                f"missing in the .npz file: {', '.join(sorted(missing))}"
            )
    base = (
        noisy_h1,
        noisy_h2,
        clean_target_data,
        mask_volume_4d,
        M_dim,
        E_dim,
        A_Dim,
        e_dim,
        ct_values,
        initial_energies,
        extra,        
    )
    
    if method == "transformation":
        special = (extra["rho"],)                    
    elif method == "normalization":
        special = (
            extra["rho"],
            extra["thetaMax"],
            extra["thetaMin"],
            extra["energyMin"],
            extra["energyMax"],
        )
    else:
        special = None 

    return base, special

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

    for m_idx in tqdm(range(noisy_input_data.shape[0]), desc="Creating CNN inputs"):
        for e_idx in range(noisy_input_data.shape[1]):
            hist_channel = noisy_input_data[m_idx, e_idx, :, :]
            energy_channel = torch.full((num_a, num_e), scaled_initial_e_vals[e_idx], dtype=torch.float32)
            ct_channel = torch.full((num_a, num_e), scaled_ct_vals[m_idx], dtype=torch.float32)
            angle_coord_channel = x_coord_grid
            final_energy_coord_channel = y_coord_grid

            combined_input_image = torch.stack([
                hist_channel, energy_channel, ct_channel, angle_coord_channel, final_energy_coord_channel
            ], dim=0)

            hist_target = noisy_target_data[m_idx, e_idx, :, :].unsqueeze(0)
            hist_mask = clean_mask_data[m_idx, e_idx, :, :].unsqueeze(0).float()

            if augment_data:
                all_channels_to_augment = torch.cat([combined_input_image, hist_target, hist_mask], dim=0)
                if np.random.rand() > 0.5:
                    all_channels_to_augment = torch.flip(all_channels_to_augment, dims=[1])
                if np.random.rand() > 0.5:
                    all_channels_to_augment = torch.flip(all_channels_to_augment, dims=[2])

                combined_input_image = all_channels_to_augment[:5]
                hist_target = all_channels_to_augment[5].unsqueeze(0)
                hist_mask = all_channels_to_augment[6].unsqueeze(0)

            X_cnn_images.append(combined_input_image)
            y_target_images.append(hist_target)
            mask_images.append(hist_mask)

    return torch.stack(X_cnn_images, dim=0), torch.stack(y_target_images, dim=0), torch.stack(mask_images, dim=0)


# --- 4. Robust U-Net Architecture with Residual, Attention Blocks, and Dropout ---
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

        print("PyTorch U-Net Model:")
        print(self.model)
        print("-" * 30)

    def train_model(self, train_loader, val_loader, num_epochs, early_stopping_patience=25):
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
                print(f'Epoch {epoch+1}/{num_epochs}, NEW BEST VAL LOSS: {val_epoch_loss:.4f}. Model saved.')
                torch.save(best_model_wts, 'best_model_weights.pth')
            else:
                epochs_without_improvement += 1  # Increment the counter
                print(f'Epoch {epoch+1}/{num_epochs}, No Improvement. Epochs without improvement: {epochs_without_improvement}')

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

        return train_losses, val_losses

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
            temp_output = outputs_scaled[i, 0, :, :] / amplitude_scaling_factor

            slice_sum = temp_output.sum()
            if slice_sum > 0:
                denoised_outputs[i, 0, :, :] = temp_output / slice_sum
            else:
                denoised_outputs[i, 0, :, :] = np.ones_like(temp_output) / (temp_output.shape[0] * temp_output.shape[1])
        return denoised_outputs


# --- 7. Data Loader Preparation Function ---
def prepare_dataloaders(X_data, y_data_noisy, mask_data, y_data_clean, batch_size, random_state=42):
    X_train, X_val, y_train_noisy, y_val_noisy, mask_train, mask_val, y_train_clean, y_val_clean = train_test_split(
        X_data, y_data_noisy, mask_data, y_data_clean,
        test_size=0.2, random_state=random_state
    )

    train_dataset = TensorDataset(X_train, y_train_noisy, mask_train)
    val_dataset = TensorDataset(X_val, y_val_noisy, mask_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
            plt.close(fig)
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

            noisy_input_img_scaled = X_full_data[flat_idx, 0, :, :].numpy()
            noisy_input_img = noisy_input_img_scaled / amplitude_scaling_factor

            single_input_tensor = X_full_data[flat_idx:flat_idx+1, :, :, :]

            predicted_clean_img = denoising_model.denoise_histogram(single_input_tensor)[0, 0, :, :]

            true_clean_img_scaled = clean_full_data[m_idx_to_plot, e_idx_to_plot, :, :].numpy()
            true_clean_img = true_clean_img_scaled / amplitude_scaling_factor

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

# --- 9. Function to Save Denoised Histograms ---
def save_denoised_histograms(denoising_model, X_full_data, output_filename, original_shape,
        hu_values, initial_energies, 
        method, special=None, extra=None, batch_size=30):
    """
    Denoises the full dataset and writes the denoised histograms **together
    with the method‑specific variables** to an NPZ file.

    Args:
        denoising_model (DenoisingModel):  The trained denoising model.
        X_full_data     (torch.Tensor):      All input data for the model.
        output_filename (str):               Name of the NPZ file to write.
        original_shape  (tuple):             The 4‑D shape (M, E, A, e)
                                            that the output must be reshaped to.
        hu_values       (list):              List that contains the
                                            histogram unit values   
        initial_energies(list):             List that contains the
                                            initial energies
        method          (str):               Either 'transformation' or
                                            'normalization'.
        special         (tuple or list):     Tuple that contains only the
                                            method‑specific arrays
                                            (e.g. (rho,) or
                                            (rho, thetaMax, thetaMin,
                                             energyMin, energyMax)).
        extra           (dict, optional):   Dictionary that contains the
                                            same method‑specific arrays
                                            (used only if *special* is
                                            None).
        batch_size      (int):               Batch size for the DataLoader.
    """
    print(f"\nDenoising full dataset and saving to {output_filename}…")
    
    # Create a DataLoader for the full dataset
    full_dataset = TensorDataset(X_full_data)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    
    denoised_outputs = []
    
    denoising_model.model.eval()
    with torch.no_grad():
        for inputs in full_loader:
            batch_denoised = denoising_model.denoise_histogram(inputs[0])
            denoised_outputs.append(batch_denoised)

    # Concatenate all denoised batches
    all_denoised = np.concatenate(denoised_outputs, axis=0)

    # Reshape the data to the original 4D structure
    denoised_4d = all_denoised.reshape(original_shape)

    # Add method‑specific keys
    payload = {"probTable": denoised_4d}
    payload = {"HU" : hu_values, "energies" : initial_energies}

    if method == "transformation":
        if special is None:
            if extra is None or "rho" not in extra:
                raise ValueError(
                    "For the 'transformation' method you must supply "
                    "'special' (or 'extra' with a 'rho' key).")
            rho = extra["rho"]
        else:
            rho = special[0]
        payload["rho"] = rho

    elif method == "normalization":
        # Expect five arrays in the special tuple
        if special is None:
            if extra is None:
                raise ValueError(
                    "For the 'normalization' method you must supply "
                    "'special' (or 'extra' with the five keys).")
            keys = ["rho", "thetaMax", "thetaMin", "energyMin", "energyMax"]
            payload.update({k: extra[k] for k in keys})
        else:
            if len(special) != 5:
                raise ValueError(
                    f"Special tuple for 'normalization' must contain "
                    f"5 arrays, got {len(special)}.")
            payload.update({
                "rho":        special[0],
                "thetaMax":   special[1],
                "thetaMin":   special[2],
                "energyMin":  special[3],
                "energyMax":  special[4]
            })
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'transformation' "
                         f"or 'normalization'.")

    # ---- 6. Persist everything into a single NPZ file ---------
    np.savez(output_filename, **payload)

    print(f"Denoised 4‑D data (and method‑specific variables) written to {output_filename}")

    
# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Train a denoising CNN with either dummy synthetic data or real data from an NPZ file.")
    data_source_group = parser.add_mutually_exclusive_group(required=True)
    data_source_group.add_argument("--dummy", action="store_true",
                                   help="Use dummy synthetic data for training.")
    
    data_source_group.add_argument('--transformation', action='store_true', help="Use transformation method")
    data_source_group.add_argument('--normalization', action='store_true', help="Use normalization method")
    # Optional: allow the user to supply a custom NPZ file path
    parser.add_argument(
        "--npz",
        type=str,
        default=None,
        help="Optional path to a custom NPZ file. Ignored if --dummy is chosen."
    )

    args = parser.parse_args()
    
    # 2. Data Loading and Initial Setup
    noisy1_histograms = None
    noisy2_target = None
    clean_target_data_proxy = None
    mask_volume_4d = None
    ct_values = None
    initial_energies = None
    
    rho, thetaMax, thetaMin, energyMin, energyMax = None, None, None, None, None
    method, special, extra = None, None, None

    if args.dummy:
        print("[INFO] Using dummy synthetic data.")
        (noisy1_histograms, noisy2_target, clean_target_data_proxy, mask_volume_4d,
         num_materials, num_initial_energies, num_angles, num_final_energies,
         ct_values, initial_energies) = generate_synthetic_noisy_data(
             num_materials, num_initial_energies, num_angles, num_final_energies,
             x_angles_range, y_final_energies_range,
             poisson_scaling_factor=100000
        )
    else:
        method = 'transformation' if args.transformation else 'normalization'
        if args.npz:
            npz_path = args.npz
        else:
            npz_path = (
                './DenoisingDataTransSheet.npz'
                if method == 'transformation'
                else './DenoisingDataNormSheet.npz'
            )
        if not os.path.isfile(npz_path):
            print(f"[ERROR] NPZ file '{npz_path}' does not exist.", file=sys.stderr)
            sys.exit(1)
            
        print(f"[INFO] Using real data from NPZ file: {npz_path}")
        base, special = load_data_from_npz(method=method, npz_path=npz_path)
        
        (
            noisy1_histograms, noisy2_target, clean_target_data_proxy, mask_volume_4d, 
            num_materials, num_initial_energies, num_angles, num_final_energies, 
            ct_values, initial_energies,
            extra  # dictionary with all method‑specific keys (rho, thetaMax, …)
        ) = base
        
        if method == 'transformation':
            rho = special[0]
            print('[INFO] Loaded 1 method-specific key: rho for transformation method')
        else:
            (
                rho, thetaMax, thetaMin, energyMin, energyMax
            ) = special
            print('[INFO] Loaded 4 method-specific keys: rho, thetaMax, thetaMin, energyMin, energyMax for normalization method')
     
    # Store the original 4D shape for reshaping the final output
    original_4d_shape = noisy1_histograms.shape
    print(f"[INFO] Original 4D shape: {original_4d_shape}")

    # Define indices to plot for visualization
    num_materials_to_plot = min(num_materials, 3)
    num_initial_energies_to_plot = min(num_initial_energies, 3)
    m_indices_to_plot_final = np.linspace(0, num_materials - 1, num_materials_to_plot, dtype=int)
    e_indices_to_plot_final = np.linspace(0, num_initial_energies - 1, num_initial_energies_to_plot, dtype=int)

    # Create the directory for saving plots
    directory = "./plots"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 3. Data Scaling and Preprocessing
    # The amplitude scaling is applied to bring values into a stable range for the model
    noisy1_histograms = noisy1_histograms * amplitude_scaling_factor
    noisy2_target = noisy2_target * amplitude_scaling_factor
    clean_target_data_proxy = clean_target_data_proxy * amplitude_scaling_factor

    X_data_torch, y_data_torch_noisy_target, mask_data_torch = create_cnn_inputs_pytorch(
        noisy1_histograms, noisy2_target, mask_volume_4d,
        ct_values, initial_energies, num_angles, num_final_energies, augment_data=False
    )
    y_data_torch_clean_reshaped = clean_target_data_proxy.reshape(-1, 1, num_angles, num_final_energies)

    print(f"[INFO] Input data shape: {X_data_torch.shape}, Target shape: {y_data_torch_noisy_target.shape}")
    print("-" * 30)

    # 4. DataLoader Preparation
    batch_size = 30
    train_loader, val_loader, _, _ = prepare_dataloaders(
        X_data_torch, y_data_torch_noisy_target, mask_data_torch, y_data_torch_clean_reshaped, batch_size
    )
    print(f"[INFO] Number of batches in train_loader: {len(train_loader)}")
    print(f"[INFO] Number of batches in val_loader: {len(val_loader)}")

    # 5. Model Initialization and Training
    model_params = {
        "in_channels": 5,
        "out_channels": 1,
        "features_list": [8, 16, 32, 64]
    }
    training_params = {
        "signal_weight": 10.0, "background_weight": 10.0, "signal_threshold": 1e-5 * amplitude_scaling_factor,
        "learning_rate": 0.00001, "weight_decay": 1e-5, "num_epochs": 200,
        "scheduler_factor": 0.1, "scheduler_patience": 5
    }

    denoising_model = DenoisingModel(model_params, training_params)
    print('Number of parameters in the model:', sum(p.numel() for p in denoising_model.model.parameters() if p.requires_grad))

    train_losses, val_losses = denoising_model.train_model(
        train_loader, val_loader, training_params['num_epochs'], early_stopping_patience=15
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
    print("Training complete. Loss plots saved.")

    # 7. Visualize Denoising Results
    plot_denoising_results(
        denoising_model,
        X_data_torch,
        clean_target_data_proxy,
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

    if method is not None:
        save_denoised_histograms(
            denoising_model=denoising_model,
            X_full_data=X_data_torch,
            output_filename="denoised_histograms.npz",
            original_shape=original_4d_shape,
            hu_values=ct_values,
            initial_energies=initial_energies,
            method=method,
            special=special,
            extra=extra,
            batch_size=batch_size
        )

    print("\nRefactored U-Net based CNN for Noise2Noise denoising is complete.")