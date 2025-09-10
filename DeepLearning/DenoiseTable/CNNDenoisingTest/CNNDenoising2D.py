import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal # For generating more structured dummy data

# --- 1. Define Dummy Data Dimensions ---
# These represent the number of discrete bins/values for each dimension
num_materials = 10       # M: Number of different materials
num_initial_energies = 10 # E: Number of different initial energy levels
num_angles = 100        # a: Number of bins for scattered angle
num_final_energies = 100 # e: Number of bins for final energy

print(f"Dataset dimensions:")
print(f"  Materials (M): {num_materials}")
print(f"  Initial Energies (E): {num_initial_energies}")
print(f"  Scattered Angles (a): {num_angles}")
print(f"  Final Energies (e): {num_final_energies}")
print("-" * 30)

# Define the actual ranges for angles and final energies for plotting and data generation
x_angles_range = np.linspace(-1, 1, num_angles) # Normalized angle range (-1 to 1)
y_final_energies_range = np.linspace(0, 100, num_final_energies) # Final energy range (0 to 100)

# --- NEW: Amplitude Scaling Factor ---
amplitude_scaling_factor = 1000.0 # Factor to multiply probabilities by during training
print(f"Amplitude Scaling Factor: {amplitude_scaling_factor}")
print("-" * 30)

# --- 2. Data Generation Function ---

def generate_synthetic_noisy_data(M_dim, E_dim, A_Dim, e_dim, x_angles_vals, y_final_energies_vals, poisson_scaling_factor=1000):
    """
    Generates synthetic 4D noisy (noisy_h1, noisy_h2) and clean histogram data for testing.
    The clean data will have a more circular/Gaussian-like peak in (a,e) space that shifts
    based on M and E indices. Poisson noise is added only where there is signal.

    Args:
        M_dim (int): Dimension for Material.
        E_dim (int): Dimension for Initial Energy.
        A_Dim (int): Dimension for Scattered Angle.
        e_dim (int): Dimension for Final Energy.
        x_angles_vals (np.ndarray): Array of actual angle values (for meshgrid).
        y_final_energies_vals (np.ndarray): Array of actual final energy values (for meshgrid).
        poisson_scaling_factor (float): Factor to scale probability data before adding Poisson noise.
                                        Higher values mean less relative noise.

    Returns:
        tuple: (noisy_h1, noisy_h2_input_to_model, clean_target_data, mask_volume_4d,
                original_M, original_E, original_a, original_e,
                ct_values, initial_energies, a_grid, e_grid)
                as PyTorch tensors and integers.
    """
    print(f"Generating synthetic data with dimensions: M={M_dim}, E={E_dim}, A={A_Dim}, e={e_dim}")

    # Dummy CT values for each material (M)
    ct_values = np.linspace(-1000, 3000, M_dim)
    print(f"Dummy CT values for materials: {ct_values}")

    # Dummy initial energy values for each energy index (E)
    initial_energies = np.linspace(10, 100, E_dim)
    print(f"Dummy Initial Energy values: {initial_energies}")

    clean_target_data = torch.zeros((M_dim, E_dim, A_Dim, e_dim), dtype=torch.float32)

    # Create 2D meshgrid for (a,e) using the provided value ranges
    a_grid = torch.tensor(x_angles_vals).float()
    e_grid = torch.tensor(y_final_energies_vals).float()
    
    a_mesh, e_mesh = torch.meshgrid(a_grid, e_grid, indexing='ij')
    pos = torch.empty(a_mesh.shape + (2,), dtype=torch.float32)
    pos[:, :, 0] = a_mesh
    pos[:, :, 1] = e_mesh

    # --- Customization Parameters for Probability Distribution Shape ---
    # Mean shift parameters - now directly related to the plot extents
    # Adding a small padding so the peak doesn't go exactly to the edge
    angle_padding_abs = 0.1 * (x_angles_vals.max() - x_angles_vals.min())
    angle_mean_min = x_angles_vals.min() + angle_padding_abs
    angle_mean_max = x_angles_vals.max() - angle_padding_abs

    energy_padding_abs = 0.1 * (y_final_energies_vals.max() - y_final_energies_vals.min())
    final_energy_mean_min = y_final_energies_vals.min() + energy_padding_abs
    final_energy_mean_max = y_final_energies_vals.max() - energy_padding_abs

    # Covariance (spread) parameters
    base_cov_angle = 0.15 # Base spread in scattered angle
    angle_cov_variation = 0.05 # How much angle spread changes with E (e.g., tighter for higher E)
    base_cov_final_energy = 10 # Base spread in final energy
    final_energy_cov_variation = 5 # How much final energy spread changes with M (e.g., tighter for denser M)
    # --- End Customization Parameters ---

    print("\nGenerating clean 4D histogram data with M and E dependencies...")

    # Iterate through M and E dimensions to place shifting Gaussian peaks
    for m_idx in range(M_dim):
        for e_idx in range(E_dim):
            # Normalize M and E indices to [0, 1] range
            norm_m = m_idx / (M_dim - 1) if M_dim > 1 else 0.5
            norm_e = e_idx / (E_dim - 1) if E_dim > 1 else 0.5

            # Determine peak center (mean) in (a,e) space based on normalized M and E
            # Peak moves right with E (initial energy), and up with M (material/CT)
            mean_angle = angle_mean_min + (angle_mean_max - angle_mean_min) * norm_e
            mean_final_energy = final_energy_mean_min + (final_energy_mean_max - final_energy_mean_min) * norm_m

            # Covariance (spread) of the distribution
            cov_angle = base_cov_angle - angle_cov_variation * norm_e
            cov_final_energy = base_cov_final_energy - final_energy_cov_variation * norm_m
            
            # Ensure positive definite covariance matrix and reasonable minimum spread
            cov_angle = max(0.01, cov_angle)
            cov_final_energy = max(1.0, cov_final_energy)

            # Create a 2D Gaussian distribution for the CLEAN data
            # scipy.stats.multivariate_normal expects numpy arrays for mean and cov
            rv = multivariate_normal(mean=[mean_angle.item(), mean_final_energy.item()], 
                                     cov=[[cov_angle**2, 0], [0, cov_final_energy**2]]) 
            
            # Evaluate the PDF over the (a,e) grid
            # Convert pos to numpy for scipy, then back to torch
            pdf_values_clean_np = rv.pdf(pos.numpy())
            pdf_values_clean = torch.from_numpy(pdf_values_clean_np).float()
            
            # Normalize the current (a,e) slice such that its sum is 1
            slice_sum = pdf_values_clean.sum()
            if slice_sum > 0:
                current_slice = pdf_values_clean / slice_sum
            else:
                current_slice = torch.ones((A_Dim, e_dim), dtype=torch.float32) / (A_Dim * e_dim) # Fallback
            
            clean_target_data[m_idx, e_idx, :, :] = current_slice

    # Create a mask for where the clean data is non-zero (or above a very small threshold)
    # This mask will determine where noise is applied
    non_zero_mask = (clean_target_data > 1e-6).float() 

    # Generate noisy_h1 and noisy_h2 using Poisson noise where signal exists
    # Scale clean data to represent "counts" before applying Poisson noise
    clean_scaled_for_poisson = clean_target_data * poisson_scaling_factor

    # Apply Poisson noise. The output of torch.poisson is float.
    # We use .round() to ensure integer counts before applying Poisson, then convert back to float.
    # This is crucial for realistic Poisson noise which operates on discrete counts.
    poisson_noise1 = torch.poisson(torch.round(clean_scaled_for_poisson))
    poisson_noise2 = torch.poisson(torch.round(clean_scaled_for_poisson * 1.1)) # Slightly different intensity for noisy2

    # Convert back to probability distribution by dividing by scaling factor
    # And apply the non-zero mask to ensure noise is only where signal was
    noisy_h1 = (poisson_noise1 / poisson_scaling_factor) * non_zero_mask
    noisy_h2_input_to_model = (poisson_noise2 / poisson_scaling_factor) * non_zero_mask

    # Ensure non-negativity (Poisson output is already non-negative, but good practice)
    noisy_h1 = torch.clamp(noisy_h1, min=0.0)
    noisy_h2_input_to_model = torch.clamp(noisy_h2_input_to_model, min=0.0)

    # --- Normalize noisy_h1 and noisy_h2 per (M, E) slice ---
    for m_idx in range(M_dim):
        for e_idx in range(E_dim):
            # Normalize noisy_h1 slice
            slice_sum_h1 = noisy_h1[m_idx, e_idx, :, :].sum()
            if slice_sum_h1 > 0:
                noisy_h1[m_idx, e_idx, :, :] /= slice_sum_h1
            else: # Fallback if sum is zero after noise (unlikely but robust)
                noisy_h1[m_idx, e_idx, :, :] = torch.ones((A_Dim, e_dim), dtype=torch.float32) / (A_Dim * e_dim)

            # Normalize noisy_h2 slice
            slice_sum_h2 = noisy_h2_input_to_model[m_idx, e_idx, :, :].sum()
            if slice_sum_h2 > 0:
                noisy_h2_input_to_model[m_idx, e_idx, :, :] /= slice_sum_h2
            else: # Fallback if sum is zero after noise
                noisy_h2_input_to_model[m_idx, e_idx, :, :] = torch.ones((A_Dim, e_dim), dtype=torch.float32) / (A_Dim * e_dim)
    # --- End Normalization ---

    original_M, original_E, original_a, original_e = noisy_h1.shape

    # Create mask based on non-zero values in the clean_target_data
    mask_volume_4d = (clean_target_data > 1e-6).bool() 

    print(f"Synthetic Noisy Volume 1 (4D) Shape: {noisy_h1.shape}")
    print(f"Synthetic Noisy Volume 2 (4D) Shape: {noisy_h2_input_to_model.shape}")
    print(f"Synthetic Clean Target Volume (4D) Shape: {clean_target_data.shape}")
    print(f"Synthetic Mask Volume (4D) Shape: {mask_volume_4d.shape}")

    return noisy_h1, noisy_h2_input_to_model, clean_target_data, mask_volume_4d, M_dim, E_dim, A_Dim, e_dim, ct_values, initial_energies, a_grid, e_grid

# --- 3. Data Transformation Function ---

def create_cnn_inputs_pytorch(noisy_input_data, noisy_target_data, clean_mask_data, ct_vals, initial_e_vals, num_a, num_e):
    """
    Transforms the 4D histogram data into a list of 5-channel 2D images (PyTorch tensors)
    for input (noisy1), and 1-channel 2D images for target (noisy2), along with a mask.
    Input Channels: [Histogram (noisy1), Energy_Scaled, CT_Scaled, Angle_Coord, Energy_Coord]

    Args:
        noisy_input_data (torch.Tensor): The 4D histogram array [N_images, a, e] for the first channel (noisy1).
        noisy_target_data (torch.Tensor): The 4D histogram array [N_images, a, e] for the target (noisy2).
        clean_mask_data (torch.Tensor): The 4D boolean mask derived from clean data.
        ct_vals (np.ndarray): 1D array of CT values for each material M.
        initial_e_vals (np.ndarray): 1D array of initial energy values for each E.
        num_a (int): Number of angle bins.
        num_e (int): Number of final energy bins.

    Returns:
        tuple: (X_data_torch, y_data_torch, mask_data_torch)
        X_data_torch (torch.Tensor): A 4D tensor of transformed input images [N_images, 5, num_a, num_e].
        y_data_torch (torch.Tensor): A 4D tensor of target images [N_images, 1, num_a, num_e].
        mask_data_torch (torch.Tensor): A 4D tensor of masks [N_images, 1, num_a, num_e].
    """
    X_cnn_images = []
    y_target_images = []
    mask_images = []
    
    # Min-Max scaling for initial energies and CT values
    min_e, max_e = np.min(initial_e_vals), np.max(initial_e_vals)
    min_ct, max_ct = np.min(ct_vals), np.max(ct_vals)

    scaled_initial_e_vals = (initial_e_vals - min_e) / (max_e - min_e) if (max_e - min_e) > 0 else np.zeros_like(initial_e_vals)
    scaled_ct_vals = (ct_vals - min_ct) / (max_ct - min_ct) if (max_ct - min_ct) > 0 else np.zeros_like(ct_vals)

    # Create normalized coordinate grids (positional encoding)
    x_coords = torch.linspace(0, 1, num_a).float() # Normalized from 0 to 1
    y_coords = torch.linspace(0, 1, num_e).float() # Normalized from 0 to 1

    x_coord_grid = x_coords.view(num_a, 1).expand(num_a, num_e)
    y_coord_grid = y_coords.view(1, num_e).expand(num_a, num_e)

    for m_idx in range(noisy_input_data.shape[0]): # Iterate through materials (M)
        for e_idx in range(noisy_input_data.shape[1]): # Iterate through initial energies (E)
            # Input Channels for X_data_torch
            hist_channel = noisy_input_data[m_idx, e_idx, :, :] # Noisy1 for input
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
            X_cnn_images.append(combined_input_image)

            # Target for y_data_torch (noisy2)
            y_target_images.append(noisy_target_data[m_idx, e_idx, :, :].unsqueeze(0)) # Add channel dimension

            # Mask for mask_data_torch (from clean data)
            mask_images.append(clean_mask_data[m_idx, e_idx, :, :].unsqueeze(0).float()) # Convert bool to float

    return torch.stack(X_cnn_images, dim=0), torch.stack(y_target_images, dim=0), torch.stack(mask_images, dim=0)


# --- 4. U-Net Architecture (DenoisingUNet Class) ---

class DenoisingUNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, features_list=None):
        super(DenoisingUNet, self).__init__()

        if features_list is None:
            features_list = [64, 128, 256, 512] # Default feature list

        # Encoder Path
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, features_list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(features_list[0]),
            nn.ReLU(True),
            nn.Conv2d(features_list[0], features_list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(features_list[0]),
            nn.ReLU(True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(features_list[0], features_list[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(features_list[1]),
            nn.ReLU(True),
            nn.Conv2d(features_list[1], features_list[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(features_list[1]),
            nn.ReLU(True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(features_list[1], features_list[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(features_list[2]),
            nn.ReLU(True),
            nn.Conv2d(features_list[2], features_list[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(features_list[2]),
            nn.ReLU(True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features_list[2], features_list[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(features_list[3]),
            nn.ReLU(True),
            nn.Conv2d(features_list[3], features_list[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(features_list[3]),
            nn.ReLU(True)
        )

        # Decoder Path
        self.upconv1 = nn.ConvTranspose2d(features_list[3], features_list[2], kernel_size=2, stride=2, output_padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(features_list[2] + features_list[2], features_list[2], kernel_size=3, padding=1), 
            nn.BatchNorm2d(features_list[2]),
            nn.ReLU(True),
            nn.Conv2d(features_list[2], features_list[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(features_list[2]),
            nn.ReLU(True)
        )

        self.upconv2 = nn.ConvTranspose2d(features_list[2], features_list[1], kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(features_list[1] + features_list[1], features_list[1], kernel_size=3, padding=1), 
            nn.BatchNorm2d(features_list[1]),
            nn.ReLU(True),
            nn.Conv2d(features_list[1], features_list[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(features_list[1]),
            nn.ReLU(True)
        )

        self.upconv3 = nn.ConvTranspose2d(features_list[1], features_list[0], kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(features_list[0] + features_list[0], features_list[0], kernel_size=3, padding=1), 
            nn.BatchNorm2d(features_list[0]),
            nn.ReLU(True),
            nn.Conv2d(features_list[0], features_list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(features_list[0]),
            nn.ReLU(True)
        )

        self.final_conv = nn.Conv2d(features_list[0], out_channels, kernel_size=1)
        self.output_activation = nn.ReLU()

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        pool1_out = self.pool1(enc1_out)

        enc2_out = self.enc2(pool1_out)
        pool2_out = self.pool2(enc2_out)

        enc3_out = self.enc3(pool2_out)
        pool3_out = self.pool3(enc3_out)

        # Bottleneck
        bottleneck_out = self.bottleneck(pool3_out)

        # Decoder with skip connections
        up1 = self.upconv1(bottleneck_out)
        dec1_in = torch.cat([up1, enc3_out], dim=1)
        dec1_out = self.dec1(dec1_in)

        up2 = self.upconv2(dec1_out)
        dec2_in = torch.cat([up2, enc2_out], dim=1)
        dec2_out = self.dec2(dec2_in)

        up3 = self.upconv3(dec2_out)
        dec3_in = torch.cat([up3, enc1_out], dim=1)
        dec3_out = self.dec3(dec3_in)

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
    def __init__(self, in_channels, out_channels, features_list, signal_weight, background_weight, signal_threshold, learning_rate):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = DenoisingUNet(in_channels, out_channels, features_list).to(self.device)
        self.criterion = MaskedMSELoss(signal_weight, background_weight, signal_threshold)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print("PyTorch U-Net Model:")
        print(self.model)
        print("-" * 30)

    def train_model(self, train_loader, val_loader, num_epochs):
        print("Starting model training...")
        train_losses = []
        val_losses = []
        train_maes = []
        val_maes = []

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            running_mae = 0.0
            for inputs, targets, masks in train_loader:
                inputs, targets, masks = inputs.to(self.device), targets.to(self.device), masks.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, masks)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_mae += torch.mean(torch.abs(outputs - targets)).item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_mae = running_mae / len(train_loader.dataset)
            train_losses.append(epoch_loss)
            train_maes.append(epoch_mae)

            self.model.eval()
            val_running_loss = 0.0
            val_running_mae = 0.0
            with torch.no_grad():
                for inputs, targets, masks in val_loader:
                    inputs, targets, masks = inputs.to(self.device), targets.to(self.device), masks.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets, masks)
                    
                    val_running_loss += loss.item() * inputs.size(0)
                    val_running_mae += torch.mean(torch.abs(outputs - targets)).item() * inputs.size(0)
                    
            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            val_epoch_mae = val_running_mae / len(val_loader.dataset)
            val_losses.append(val_epoch_loss)
            val_maes.append(val_epoch_mae)

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train MAE: {epoch_mae:.4f}, '
                  f'Val Loss: {val_epoch_loss:.4f}, Val MAE: {val_epoch_mae:.4f}')

        print("Model training finished.")
        print("-" * 30)
        return train_losses, val_losses, train_maes, val_maes

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
            # Scale back down
            temp_output = outputs_scaled[i, 0, :, :] / amplitude_scaling_factor
            
            # Then normalize to sum to 1
            slice_sum = temp_output.sum()
            if slice_sum > 0:
                denoised_outputs[i, 0, :, :] = temp_output / slice_sum
            else:
                denoised_outputs[i, 0, :, :] = np.ones_like(temp_output) / (temp_output.shape[0] * temp_output.shape[1])
        return denoised_outputs

# --- 7. Data Loader Preparation Function ---

def prepare_dataloaders(X_data, y_data_noisy, mask_data, y_data_clean, batch_size, random_state=42):
    """
    Splits data into train/validation sets and creates PyTorch DataLoaders.
    """
    X_train, X_val, \
    y_train_noisy, y_val_noisy, \
    mask_train, mask_val, \
    y_train_clean, y_val_clean = train_test_split(
        X_data, y_data_noisy, mask_data, y_data_clean,
        test_size=0.2, random_state=random_state
    )

    train_dataset = TensorDataset(X_train, y_train_noisy, mask_train)
    val_dataset = TensorDataset(X_val, y_val_noisy, mask_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_val, y_val_clean # Return X_val and y_val_clean for specific plotting

# --- 8. Plotting Function ---

def plot_denoising_results(denoising_model, X_full_data, clean_full_data, m_indices, e_indices, ct_vals, initial_e_vals, x_angles_np, y_final_energies_np, amplitude_scaling_factor):
    """
    Plots denoising results for specific (M, E) combinations.
    """
    print("Plotting denoising results for specified (M, E) pairs...")
    plot_counter = 0
    for m_idx_to_plot in m_indices:
        for e_idx_to_plot in e_indices:
            # Get the index in the flattened (M*E) dimension for the full dataset
            flat_idx = m_idx_to_plot * num_initial_energies + e_idx_to_plot
            
            # Retrieve original noisy input (channel 0) from X_full_data (needs to be scaled back for plotting)
            noisy_input_img_scaled = X_full_data[flat_idx, 0, :, :].numpy() 
            noisy_input_img = noisy_input_img_scaled / amplitude_scaling_factor
            
            # Get the input tensor for denoising (full 5 channels)
            single_input_tensor = X_full_data[flat_idx:flat_idx+1, :, :, :]
            
            # Denoise using the model's denoise_histogram method
            predicted_clean_img = denoising_model.denoise_histogram(single_input_tensor)[0, 0, :, :]
            
            # Retrieve true clean image (from clean_full_data, needs to be scaled back for plotting)
            true_clean_img_scaled = clean_full_data[m_idx_to_plot, e_idx_to_plot, :, :].numpy()
            true_clean_img = true_clean_img_scaled / amplitude_scaling_factor

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Denoising Example (M: {ct_vals[m_idx_to_plot]:.0f} CT, E: {initial_e_vals[e_idx_to_plot]:.0f} keV)', fontsize=16)

            # Plot Noisy Input (noisy1)
            im1 = axes[0].imshow(noisy_input_img.T, origin='lower', aspect='auto', cmap='viridis',
                                 extent=[x_angles_np.min(), x_angles_np.max(), y_final_energies_np.min(), y_final_energies_np.max()])
            axes[0].set_title('Noisy Input (Poisson Noise 1)')
            axes[0].set_xlabel('Scattered Angle')
            axes[0].set_ylabel('Final Energy')
            plt.colorbar(im1, ax=axes[0], label='Probability')

            # Plot Predicted Clean
            im2 = axes[1].imshow(predicted_clean_img.T, origin='lower', aspect='auto', cmap='viridis',
                                 extent=[x_angles_np.min(), x_angles_np.max(), y_final_energies_np.min(), y_final_energies_np.max()])
            axes[1].set_title('Predicted Denoised (Noise2Noise)')
            axes[1].set_xlabel('Scattered Angle')
            axes[1].set_ylabel('Final Energy')
            plt.colorbar(im2, ax=axes[1], label='Probability')

            # Plot True Clean
            im3 = axes[2].imshow(true_clean_img.T, origin='lower', aspect='auto', cmap='viridis',
                                 extent=[x_angles_np.min(), x_angles_np.max(), y_final_energies_np.min(), y_final_energies_np.max()])
            axes[2].set_title('True Clean')
            axes[2].set_xlabel('Scattered Angle')
            axes[2].set_ylabel('Final Energy')
            plt.colorbar(im3, ax=axes[2], label='Probability')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'denoising_example_M{m_idx_to_plot}_E{e_idx_to_plot}.png', dpi=300)
            plt.close(fig)
            plot_counter += 1
            if plot_counter >=1050: # Limit to 3 plots for brevity if many M,E combinations
                break
        if plot_counter >= 100:
            break

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Data Generation ---
    (noisy1_histograms, noisy2_input_to_model, clean_target_data, mask_volume_4d,
     original_M, original_E, original_a, original_e,
     ct_values, initial_energies, a_grid, e_grid) = generate_synthetic_noisy_data(
        num_materials, num_initial_energies, num_angles, num_final_energies,
        x_angles_range, y_final_energies_range,
        poisson_scaling_factor=1000000
    )

    # Apply amplitude scaling to the generated data
    noisy1_histograms *= amplitude_scaling_factor
    noisy2_input_to_model *= amplitude_scaling_factor
    clean_target_data *= amplitude_scaling_factor

    # --- Data Transformation for CNN Input ---
    X_data_torch, y_data_torch_noisy, mask_data_torch = create_cnn_inputs_pytorch(
        noisy1_histograms, noisy2_input_to_model, mask_volume_4d,
        ct_values, initial_energies, num_angles, num_final_energies
    )

    print(f"Shape of transformed input data (X_data_torch): {X_data_torch.shape}")
    print(f"  (Number of images, Channels (5), Angles, Final Energies)")
    print(f"Shape of noisy target data (y_data_torch_noisy): {y_data_torch_noisy.shape}")
    print(f"  (Number of images, 1 Channel, Angles, Final Energies)")
    print(f"Shape of mask data (mask_data_torch): {mask_data_torch.shape}")
    print(f"  (Number of images, 1 Channel, Angles, Final Energies)")
    print("-" * 30)

    # Prepare clean data for plotting comparison (reshaped)
    y_data_torch_clean_reshaped = clean_target_data.reshape(-1, 1, num_angles, num_final_energies)

    # --- Prepare DataLoaders ---
    batch_size = 10
    train_loader, val_loader, X_val_for_plotting, y_val_clean_for_plotting = prepare_dataloaders(
        X_data_torch, y_data_torch_noisy, mask_data_torch, y_data_torch_clean_reshaped, batch_size
    )

    # --- Initialize and Train Denoising Model ---
    model_params = {
        "in_channels": 5,
        "out_channels": 1,
        "features_list": [16, 32, 64, 128],
        "signal_weight": 10000.0,
        "background_weight": 1.0,
        "signal_threshold": 1e-6 * amplitude_scaling_factor,
        "learning_rate": 0.00001
    }
    denoising_model = DenoisingModel(**model_params)
    print(denoising_model)
    print('Number of parameters in the model:', sum(p.numel() for p in denoising_model.model.parameters() if p.requires_grad))

    num_epochs = 500
    train_losses, val_losses, train_maes, val_maes = denoising_model.train_model(
        train_loader, val_loader, num_epochs
    )

    # Plot training & validation loss and MAE
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Model Loss (Weighted MSE)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(train_maes)
    plt.plot(val_maes)
    plt.title('Model Mean Absolute Error (Unweighted)')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('training_validation_loss_mae.png', dpi=300)
    plt.close()

    print("Training complete. Loss and MAE plots saved.")

    # --- Make Predictions and Plot Denoising Results for specific M, E ---
    # Define indices to plot as requested (same as initial data plotting)
    m_indices_to_plot_final = np.linspace(0, num_materials - 1, 5, dtype=int) # Plot up to 3 materials
    e_indices_to_plot_final = np.linspace(0, num_initial_energies - 1, 5, dtype=int) # Plot up to 3 initial energies

    plot_denoising_results(
        denoising_model, 
        X_data_torch, # Use full X_data_torch to get specific M,E pairs
        clean_target_data, # Use full clean_target_data for true clean
        m_indices_to_plot_final, 
        e_indices_to_plot_final, 
        ct_values, 
        initial_energies, 
        x_angles_range, 
        y_final_energies_range,
        amplitude_scaling_factor
    )

    print("\nRefactored U-Net based CNN for Noise2Noise denoising is complete.")
    print("The plots above show the noisy input, the model's denoised output, and the true clean distribution for specific (M, E) pairs.")
