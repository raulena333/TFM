import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm # For progress bars
import matplotlib.pyplot as plt # For plotting
import time # For timing training/inference

# --- 1. Define Positional Encoding and INR Model ---

class PositionalEncoding(nn.Module):
    def __init__(self, num_input_dims, L=10):
        super().__init__()
        self.L = L # Number of frequencies
        self.funcs = [torch.sin, torch.cos]
        # Calculate the output dimension: original dims + (sin + cos) * L for each dim
        self.num_output_dims = num_input_dims * (len(self.funcs) * L + 1)

    def forward(self, x):
        # x is (Batch, num_input_dims) e.g., (Batch, 4) for (M, E, A, B)
        if self.L == 0: # If L is 0, no encoding, just pass through
            return x
            
        # Create a list to hold all encoded components
        encodings = [x] # Start with the original coordinates
        
        # For each frequency 'l' and each function (sin/cos)
        # apply the transformation to all input dimensions in 'x'
        for l in range(self.L):
            for func in self.funcs:
                # 2.**l * np.pi * x creates scaled versions of input coordinates
                # e.g., x, 2pi*x, 4pi*x, 8pi*x, ...
                encodings.append(func(2.**l * np.pi * x)) 
        
        # Concatenate all generated encodings along the last dimension (feature dimension)
        return torch.cat(encodings, dim=-1)


class INRModel(nn.Module):
    def __init__(self, input_coords_dim=4, output_value_dim=1,
                 hidden_dim=256, num_layers=8, positional_encoding_L=6):
        super().__init__()

        # Initialize the positional encoder
        self.pos_encoder = PositionalEncoding(input_coords_dim, L=positional_encoding_L)
        # Calculate the dimension of the input after positional encoding
        current_dim = self.pos_encoder.num_output_dims

        # Build the Multi-Layer Perceptron (MLP)
        layers = []
        # First layer: maps from encoded coordinates to hidden_dim
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.SiLU()) # Changed from nn.ReLU to nn.SiLU
        
        # Subsequent hidden layers
        for _ in range(num_layers - 1): # num_layers-1 because the first layer is already added
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU()) # Changed from nn.ReLU to nn.SiLU
        
        # Final layer: maps from hidden_dim to the desired output_value_dim (1 for probability)
        layers.append(nn.Linear(hidden_dim, output_value_dim))
        
        # Combine all layers into a sequential module
        self.net = nn.Sequential(*layers) 
        
        # Final activation to ensure output is in a valid range (0 to 1 for probability)
        self.final_activation = nn.Sigmoid() 

    def forward(self, coords):
        # coords shape: (Batch, 4) for (M, E, A, B)
        # First, apply positional encoding to the input coordinates
        encoded_coords = self.pos_encoder(coords)
        # Pass the encoded coordinates through the MLP
        output = self.net(encoded_coords)
        # Apply the final activation if defined
        if self.final_activation:
            output = self.final_activation(output)
        return output

# --- 2. Custom Dataset for INR Training ---

class HistogramINRPairsDataset(Dataset):
    def __init__(self, noisy_data_1, noisy_data_2):
        """
        Creates a dataset of (coordinates, noisy_value_1, noisy_value_2) tuples
        from the 4D histogram data for INR training.

        Args:
            noisy_data_1 (np.ndarray): First noisy 4D histogram (M, E, A, B).
            noisy_data_2 (np.ndarray): Second noisy 4D histogram (M, E, A, B).
        """
        if noisy_data_1.shape != noisy_data_2.shape:
            raise ValueError("Noisy data 1 and 2 must have the same shape.")

        self.shape = noisy_data_1.shape
        M, E, A, B = self.shape

        # Generate all possible coordinates for the 4D grid
        # Normalize coordinates to [-1, 1] for better model performance with SiLU
        coords_m = torch.linspace(-1, 1, M, dtype=torch.float32) # Changed from 0 to 1
        coords_e = torch.linspace(-1, 1, E, dtype=torch.float32) # Changed from 0 to 1
        coords_a = torch.linspace(-1, 1, A, dtype=torch.float32) # Changed from 0 to 1
        coords_b = torch.linspace(-1, 1, B, dtype=torch.float32) # Changed from 0 to 1

        # Create a meshgrid of all 4D coordinates
        # indexing='ij' ensures (M,E,A,B) order for the meshgrid output
        mesh_m, mesh_e, mesh_a, mesh_b = torch.meshgrid(coords_m, coords_e, coords_a, coords_b, indexing='ij')

        # Stack them into a (num_total_points, 4) tensor
        self.all_coords = torch.stack([mesh_m.flatten(), 
                                       mesh_e.flatten(), 
                                       mesh_a.flatten(), 
                                       mesh_b.flatten()], dim=1) # Shape: (M*E*A*B, 4)

        # Flatten the histogram values and convert to PyTorch tensors
        self.noisy_values_1 = torch.from_numpy(noisy_data_1.flatten()).unsqueeze(1) # Shape: (M*E*A*B, 1)
        self.noisy_values_2 = torch.from_numpy(noisy_data_2.flatten()).unsqueeze(1) # Shape: (M*E*A*B, 1)

        print(f"Dataset created with {len(self.all_coords)} coordinate-value pairs.")

    def __len__(self):
        # The total number of data points in the dataset
        return len(self.all_coords)

    def __getitem__(self, idx):
        # Return (coordinates, noisy_value_1, noisy_value_2) for a given index
        return self.all_coords[idx], self.noisy_values_1[idx], self.noisy_values_2[idx]


# --- 3. Provided Data Loading Function ---
def load_and_prepare_data(npz_path="./DenoisingDataTransSheet.npz"):
    """
    Loads histogram data from an NPZ file and prepares noisy and ground truth tensors.
    Ensures data is float32.

    Args:
        npz_path (str): Path to the .npz file containing histogram data.
                        Expected to have a key "histograms" with shape (2, M, E, A, B).
                        histograms[0] is treated as noisy input, histograms[1] as ground truth.

    Returns:
        tuple: A tuple containing (noisy_histograms_np, ground_truth_histograms_np).
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
    
    print(f"Loaded noisy_input_data shape: {noisy_input_data.shape}")
    
    print(f"Noisy volume 4D shape (M,E,A,B): {noisy_input_data.shape}")
    print(f"Target volume 4D shape (M,E,A,B): {target_data.shape}")

    return noisy_input_data, target_data

# --- 4. Plotting Function ---
def plot_4d_histogram_slice(
    data_noisy, data_denoised, 
    fixed_m_idx, fixed_e_idx, 
    title_suffix="", 
    angleRange = (0, 70), energyRange = (-0.6, 0) 
):
    """
    Plots a 2D slice (Angle vs Final Energy) from a 4D histogram.

    Args:
        data_noisy (np.ndarray or torch.Tensor): The original noisy 4D histogram (M, E, A, B).
        data_denoised (np.ndarray or torch.Tensor): The denoised 4D histogram (M, E, A, B).
        fixed_m_idx (int): The index for the M (Material) dimension to slice.
        fixed_e_idx (int): The index for the E (Initial Energy) dimension to slice.
        title_suffix (str): An additional string to append to the plot titles.
    """
    # Ensure data is NumPy array for plotting
    data_noisy_np = data_noisy.cpu().numpy() if isinstance(data_noisy, torch.Tensor) else data_noisy
    data_denoised_np = data_denoised.cpu().numpy() if isinstance(data_denoised, torch.Tensor) else data_denoised

    # Extract the 2D slices
    # Slice shape will be (A, B)
    slice_noisy = data_noisy_np[fixed_m_idx, fixed_e_idx, :, :]
    slice_denoised = data_denoised_np[fixed_m_idx, fixed_e_idx, :, :]
        
    extent = (angleRange[0], angleRange[1], energyRange[0], energyRange[1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Histogram Slices (M={fixed_m_idx}, E={fixed_e_idx}) {title_suffix}", fontsize=16)

    # Plot Noisy Slice
    im1 = axes[0].imshow(slice_noisy, cmap='viridis', origin='lower', extent=extent, aspect="auto")
    axes[0].set_title("Original Noisy Slice")
    axes[0].set_xlabel("Transformed Angle")
    axes[0].set_ylabel("Transformed Energy")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot Denoised Slice
    im2 = axes[1].imshow(slice_denoised, cmap='viridis', origin='lower', extent=extent, aspect="auto")
    axes[1].set_title("Denoised Slice (INR)")
    axes[1].set_xlabel("Transformed Angle")
    axes[1].set_ylabel("Transformed Energy")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig(f"./DenoisingExample_M{fixed_m_idx}_E{fixed_e_idx}{title_suffix}.pdf")
    plt.close()


# --- 5. Main Training and Inference Logic ---
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- Load your actual data ---
    npz_file_path = "./DenoisingDataTransSheet.npz" # Make sure this path is correct
    try:
        noisy_hist_1_np, noisy_hist_2_np = load_and_prepare_data(npz_file_path)
    except FileNotFoundError as e:
        print(e)
        print("Please ensure 'DenoisingDataTransSheet.npz' is in the correct directory.")
        exit() # Exit if data not found

    M_dim, E_dim, A_dim, B_dim = noisy_hist_1_np.shape
    print(f"Data dimensions (M, E, A, B): ({M_dim}, {E_dim}, {A_dim}, {B_dim})")

    # --- Create Dataset and DataLoader ---
    # The dataset generates (coords, noisy_val_1, noisy_val_2) tuples
    train_dataset = HistogramINRPairsDataset(noisy_hist_1_np, noisy_hist_2_np)
    
    # Using a larger batch size is typically good for INR as it speeds up training
    # Adjust based on your GPU memory. 
    # For 25M points, 8192 batch size gives ~3052 batches per epoch.
    BATCH_SIZE = 16384 # 8192 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    print(f"DataLoader created with batch_size: {BATCH_SIZE}")
    print(f"Using {os.cpu_count()} CPU workers for DataLoader")

    # --- Initialize INR Model, Optimizer, and Loss ---
    # input_coords_dim is 4 (M, E, A, B)
    # output_value_dim is 1 (the probability value)
    # positional_encoding_L=6 is a good starting point for learning details
    model_inr = INRModel(input_coords_dim=4, output_value_dim=1,
                         hidden_dim=1024, num_layers=12, positional_encoding_L=12).to(DEVICE)
    
    LEARNING_RATE = 1e-5 # Start with a slightly lower LR
    optimizer = torch.optim.Adam(model_inr.parameters(), lr=LEARNING_RATE) 
    criterion = nn.MSELoss() # Standard for Noise2Noise

    # --- Training Loop ---
    EPOCHS = 10 # Adjusted epochs for potentially better convergence with positional encoding
    print(f"\nStarting INR training for {EPOCHS} epochs...")
    start_time_training = time.time()
    
    for epoch in range(EPOCHS):
        model_inr.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for coords, val1, val2 in pbar:
            coords, val1, val2 = coords.to(DEVICE), val1.to(DEVICE), val2.to(DEVICE)

            optimizer.zero_grad()
            
            # Noise2Noise: Model takes coords, predicts the "denoised" version of val1,
            # and we compare it against val2.
            predictions = model_inr(coords)
            
            loss = criterion(predictions, val2) 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.8f}")

    end_time_training = time.time()
    print(f"\nINR training complete in {end_time_training - start_time_training:.2f} seconds.")

    # --- Inference: Denoise the Full Histogram ---
    print("\nStarting full histogram denoising using trained INR model...")
    model_inr.eval() # Set model to evaluation mode

    start_time_inference = time.time()
    with torch.no_grad():
        # Generate all coordinates for the full output histogram
        # These are normalized coordinates [-1, 1]
        coords_m = torch.linspace(-1, 1, M_dim, dtype=torch.float32).to(DEVICE)
        coords_e = torch.linspace(-1, 1, E_dim, dtype=torch.float32).to(DEVICE)
        coords_a = torch.linspace(-1, 1, A_dim, dtype=torch.float32).to(DEVICE)
        coords_b = torch.linspace(-1, 1, B_dim, dtype=torch.float32).to(DEVICE)

        mesh_m, mesh_e, mesh_a, mesh_b = torch.meshgrid(coords_m, coords_e, coords_a, coords_b, indexing='ij')
        
        all_coords_flattened = torch.stack([mesh_m.flatten(), mesh_e.flatten(), mesh_a.flatten(), mesh_b.flatten()], dim=1)
        
        # Process in batches to avoid out of memory issues, especially for large histograms
        # This batch size is for inference, can be larger than training batch size
        BATCH_SIZE_INFERENCE = 65536 
        
        denoised_histogram_flat_list = []
        
        # Use tqdm for progress bar during inference
        for i in tqdm(range(0, all_coords_flattened.shape[0], BATCH_SIZE_INFERENCE), desc="Denoising progress"):
            batch_coords = all_coords_flattened[i:i+BATCH_SIZE_INFERENCE]
            
            # Predict the denoised values for this batch of coordinates
            denoised_batch_values = model_inr(batch_coords)
            
            denoised_histogram_flat_list.append(denoised_batch_values.cpu())
        
        # Concatenate all denoised batches and reshape to the original 4D histogram shape
        denoised_histogram_flat = torch.cat(denoised_histogram_flat_list, dim=0)
        
        # Reshape back to the original 4D histogram dimensions
        # .squeeze() is important because the model outputs (N, 1) and we want (M,E,A,B)
        denoised_histogram_4d = denoised_histogram_flat.reshape(M_dim, E_dim, A_dim, B_dim).squeeze() 
        
        print(f"Full denoised histogram generated with shape: {denoised_histogram_4d.shape}")

    end_time_inference = time.time()
    print(f"Inference complete in {end_time_inference - start_time_inference:.2f} seconds.")

    # --- Calculate and print model size ---
    param_size = 0
    for param in model_inr.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model_inr.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_bytes = param_size + buffer_size
    size_all_mb = size_all_bytes / 1024**2
    print(f"Trained model size: {size_all_mb:.3f}MB")

    # --- Plotting Examples ---
    print("\nPlotting sample slices for comparison...")

    # Define a list of (M, E) index pairs you want to plot
    # Make sure these indices are within the actual dimensions of your data (0 to M_dim-1, 0 to E_dim-1)
    indices_to_plot = [
        (0, 0),    # First M, First E
        (M_dim // 4, E_dim // 4),   # ~1/4 way through M and E
        (M_dim // 2, E_dim // 2),  # Middle slice
        (M_dim * 3 // 4, E_dim * 3 // 4), # ~3/4 way through M and E
        (M_dim - 1, E_dim - 1)   # Last M, Last E
    ]

    for m_idx, e_idx in indices_to_plot:
        # Add a check to ensure indices are valid for your data
        if 0 <= m_idx < M_dim and 0 <= e_idx < E_dim:
            plot_4d_histogram_slice(
                noisy_hist_1_np, denoised_histogram_4d, 
                fixed_m_idx=m_idx, fixed_e_idx=e_idx, 
                title_suffix=f" (M={m_idx}, E={e_idx})"
            )
        else:
            print(f"Skipping plot for invalid indices: M={m_idx}, E={e_idx}. Out of bounds.")

    print("\nPlotting complete. Check the generated plots.")