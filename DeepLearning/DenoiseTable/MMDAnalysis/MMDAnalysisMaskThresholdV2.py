import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
from skimage.measure import label
from typing import Tuple, Union
import matplotlib.pylab as pylab

# Matplotlib params
params = {
    'xtick.labelsize': 15,    
    'ytick.labelsize': 15,      
    'axes.titlesize': 17,
    'axes.labelsize': 17,
    'legend.fontsize': 17
}
pylab.rcParams.update(params)  # Apply changes

# Helper functions for MMD calculation
def _rbf_kernel(x, y, gamma):
    """
    Computes the Gaussian Radial Basis Function (RBF) kernel matrix.

    Args:
        x (torch.Tensor): A tensor of shape [n_samples, n_features].
        y (torch.Tensor): A tensor of shape [m_samples, n_features].
        gamma (float): The kernel bandwidth parameter.

    Returns:
        torch.Tensor: The kernel matrix of shape [n_samples, m_samples].
    """
    x_sqnorms = torch.sum(x * x, dim=1)
    y_sqnorms = torch.sum(y * y, dim=1)
    xy_matmul = torch.matmul(x, y.T)
    distances_sq = (
        torch.unsqueeze(x_sqnorms, 1)
        + torch.unsqueeze(y_sqnorms, 0)
        - 2 * xy_matmul
    )
    # Ensure distances are non-negative
    distances_sq = torch.clamp(distances_sq, min=0.0)
    kernel_matrix = torch.exp(-gamma * distances_sq)
    return kernel_matrix

def mmd_loss(x, y, sigma):
    """
    Calculates the Maximum Mean Discrepancy (MMD) loss.

    This function measures the distance between two sets of samples, x and y,
    drawn from two distributions.

    Args:
        x (torch.Tensor): A tensor of samples from the first distribution.
        y (torch.Tensor): A tensor of samples from the second distribution.
        sigma (float): The bandwidth parameter for the RBF kernel.

    Returns:
        torch.Tensor: The squared MMD value.
    """
    x = x.float()
    y = y.float()
    gamma = 1.0 / (2 * sigma**2)
    k_xx = _rbf_kernel(x, x, gamma)
    k_yy = _rbf_kernel(y, y, gamma)
    k_xy = _rbf_kernel(x, y, gamma)
    mmd_sq = torch.mean(k_xx) + torch.mean(k_yy) - 2 * torch.mean(k_xy)
    mmd_sq = torch.clamp(mmd_sq, min=0.0)
    return mmd_sq


# --- New functions for component-based thresholding ---
def largest_component_mask(mask: np.ndarray, connectivity: int = 2) -> np.ndarray:
    """
    Returns the boolean mask of the largest connected component in 2D.
    - connectivity=1 -> 4-connectivity; connectivity=2 -> 8-connectivity
    """
    # Assert that the input is a 2D image.
    assert mask.ndim == 2, "Expected a 2D image."
    mask = mask.astype(bool)
    
    # If the mask is empty, return an empty mask.
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    # Label all connected components in the mask.
    labeled = label(mask, connectivity=connectivity)
    
    # Count the number of pixels for each labeled component.
    counts = np.bincount(labeled.ravel())
    
    # If there's only a background, return an empty mask.
    if counts.size <= 1:
        return np.zeros_like(mask, dtype=bool)
        
    # Ignore the background (label 0) by setting its count to 0.
    counts[0] = 0
    
    # Find the label of the largest component.
    largest_label = counts.argmax()
    
    # Return a new mask containing only the largest component.
    return labeled == largest_label


def keep_largest_component_from_threshold(
    img: np.ndarray,
    threshold: float,
    fill_value: Union[int, float] = 0,
    connectivity: int = 2,
    return_mask: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Applies mask = (img > threshold), keeps only the largest component, and
    returns the image with only that region. Optionally returns the mask as well.

    Parameters
    ----------
    img : np.ndarray (2D)
        The input 2D image array.
    threshold : float | int
        The value used to create the initial binary mask.
    fill_value : Union[int, float], optional
        The value to use for the background (default is 0).
    connectivity : int, optional
        The connectivity for the largest component search: 1 for 4-connectivity
        or 2 for 8-connectivity (default is 2).
    return_mask : bool, optional
        If True, the function also returns the mask of the largest component (default is False).

    Returns
    -------
    img_only : np.ndarray
        The image with only the largest connected component.
    (optional) lcc_mask : np.ndarray (bool)
        The mask of the largest component.
    """
    # Assert that the input is a 2D image.
    assert img.ndim == 2, "Expected a 2D image."

    # Create an initial binary mask based on the threshold.
    mask = (img > threshold) & np.isfinite(img)
    # Put zero if img is NaN or Inf
    mask = mask & np.isfinite(img)

    # Find the largest connected component within the mask.
    lcc_mask = largest_component_mask(mask, connectivity=connectivity)

    # Apply the largest component mask to the original image.
    img_only = np.where(lcc_mask, img, fill_value).astype(img.dtype)

    # Return the image and optionally the mask.
    return (img_only, lcc_mask) if return_mask else img_only


# Main functions for data loading and MMD calculation
def load_data(noisy_file_path, denoised_file_path):
    """
    Loads noisy and denoised data from .npz files using memory-mapping,
    along with HU and initial energy values.

    Args:
        noisy_file_path (str): The file path to the noisy data .npz file.
                               Expected keys: 'histograms', 'hu_values', 'initial_energies'
        denoised_file_path (str): The file path to the denoised data .npz file.
                                Expected key: 'probTable'
    
    Returns:
        tuple: A tuple containing the memory-mapped noisy data, denoised data,
               HU values, and initial energies, or (None, None, None, None) if an error occurs.
    """
    try:
        print("[INFO] Loading data using memory-mapping...")
        
        # Load noisy data using mmap_mode='r'
        noisy_npz = np.load(noisy_file_path, mmap_mode='r')
        noisy_data = noisy_npz['histograms']
        hu_values = noisy_npz['HU']
        initial_energies = noisy_npz['energies']
        
        # Load denoised data using mmap_mode='r'
        denoised_npz = np.load(denoised_file_path, mmap_mode='r')
        denoised_data = denoised_npz['probTable']
        
        # Note: Thresholding is now handled in save_denoised_data, but a
        # simple check is here for clarity.
        denoised_data_shape = denoised_data.shape
        noisy_data_shape = noisy_data.shape
        
        if noisy_data_shape[0] != 2:
            raise ValueError(f"Expected noisy data to be 5D with shape [2, M, E, a, e], but got {noisy_data_shape}.")
        
        if len(denoised_data_shape) != 4:
            raise ValueError(f"Expected denoised data to be 4D with shape [M, E, a, e], but got {denoised_data_shape}.")

        if not np.array_equal(noisy_data_shape[1:], denoised_data_shape):
             raise ValueError(f"The shapes of the last 4 dimensions of noisy data ({noisy_data_shape[1:]}) and denoised data ({denoised_data_shape}) do not match.")

        print(f"[INFO] Data memory-mapped successfully:")
        print(f"\t Noisy data shape: {noisy_data_shape}")
        print(f"\t Denoised data shape: {denoised_data_shape}")
        print(f"\t HU values shape: {hu_values.shape}")
        print(f"\t Initial energies shape: {initial_energies.shape}")

        return noisy_data, denoised_data, hu_values, initial_energies
        
    except FileNotFoundError as e:
        print(f"Error: One or more files not found. Skipping analysis.")
        print(f"   {e}")
        return None, None, None, None
    except KeyError as e:
        print(f"Error: A required key (e.g., 'histograms', 'hu_values', 'initial_energies', 'probTable') is missing from a .npz file. Skipping analysis.")
        print(f"   {e}")
        return None, None, None, None
    except ValueError as e:
        print(f"Error: Data has an incorrect shape. Skipping analysis.")
        print(f"   {e}")
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during data loading. Skipping analysis.")
        print(f"   {e}")
        return None, None, None, None

def feature_extractor(data):
    """
    Extracts features from the histograms by flattening them.
    
    Args:
        data (np.ndarray): Input data array of shape [..., 100, 100].
    
    Returns:
        torch.Tensor: A tensor of features with shape [num_histograms, 10000].
    """
    original_shape = data.shape
    num_histograms = np.prod(original_shape[:-2])
    
    features = data.reshape(num_histograms, -1)
    
    return torch.from_numpy(features)

def save_denoised_data(original_path, new_path, threshold):
    """
    Saves the thresholded and re-normalized denoised data back to a new .npz file,
    preserving all other arrays from the original file.

    This function applies a threshold to the 'probTable' array and then
    re-normalizes each histogram (for each M, E index) so that its
    elements sum to 1.

    Args:
        original_path (str): The path to the original denoised .npz file.
        new_path (str): The path to save the new .npz file.
        threshold (float): The threshold value to apply to the data.
    """
    try:
        print(f"\n[INFO] Saving thresholded and re-normalized denoised data to '{new_path}'...")

        # Load all data from the original file
        with np.load(original_path) as original_npz:

            # Create a dictionary to hold all arrays to be saved
            data_to_save = {}

            # Process the 'probTable' key specifically
            if 'probTable' in original_npz.files:
                denoised_data = original_npz['probTable'].copy()
                
                # Get the shape of the data
                M, E, A, e = denoised_data.shape
                
                print("[INFO] Applying threshold and keeping largest connected component...")
                for m_idx in tqdm(range(M), desc="Applying component filter"):
                    for e_idx in range(E):
                        histogram_2d = denoised_data[m_idx, e_idx, :, :]
                        
                        # --- THE TWO-STEP PROCESS ---
                        # Step 1: Threshold and find the largest component.
                        cleaned_histogram = keep_largest_component_from_threshold(
                            img=histogram_2d,
                            threshold=threshold,
                            fill_value=0
                        )

                        # Step 2: Re-normalize the cleaned histogram.
                        current_sum = np.sum(cleaned_histogram)
                        if current_sum > 0.0001:
                            denoised_data[m_idx, e_idx, :, :] = cleaned_histogram / current_sum
                        else:
                            denoised_data[m_idx, e_idx, :, :] = 0
                
                print(f'The sum of the denoised data is: {np.sum(denoised_data)}')
                data_to_save['probTable'] = denoised_data

            # Copy all other arrays from the original file
            for key in original_npz.files:
                if key != 'probTable':
                    data_to_save[key] = original_npz[key]

        # Use np.savez_compressed for better file size management
        np.savez_compressed(new_path, **data_to_save)
        print("[INFO] Denoised data with threshold and normalization saved successfully.")

    except Exception as e:
        print(f"Error: Could not save the denoised data. {e}")


def calculate_mmd_per_me(denoised_hist, noisy1_hist, noisy2_hist):
    """
    Calculates MMD for a single (M, E) pair.

    Args:
        denoised_hist (np.ndarray): The single denoised histogram, shape [100, 100].
        noisy1_hist (np.ndarray): The first noisy histogram, shape [100, 100].
        noisy2_hist (np.ndarray): The second noisy histogram, shape [100, 100].
    
    Returns:
        tuple: (mmd_denoised_vs_noisy1, mmd_denoised_vs_noisy2) for this pair.
    """
    # Each histogram becomes a single sample with 10000 features.
    features_denoised = feature_extractor(denoised_hist[np.newaxis, :])
    features_noisy1 = feature_extractor(noisy1_hist[np.newaxis, :])
    features_noisy2 = feature_extractor(noisy2_hist[np.newaxis, :])

    # To calculate MMD, we need more than one sample per "distribution".
    # The MMD between two single samples is simply the distance between them.
    # To compare the denoised histogram to the distribution of noisy ones, we
    # form a set of samples from the noisy distributions.
    features_noisy_set1 = features_noisy1
    features_noisy_set2 = features_noisy2
    
    # We must also compare the single denoised sample to the "distribution"
    # of the noisy samples. A common approach is to use the two available samples.
    features_denoised_set = features_denoised

    # We need to choose a 'sigma' for the RBF kernel.
    # We use the median distance between all samples in this specific (M,E) pair.
    all_features = torch.cat([features_denoised, features_noisy1, features_noisy2], dim=0)
    dist_matrix = torch.cdist(all_features, all_features)
    median_dist = torch.median(dist_matrix[dist_matrix > 0]) # Exclude self-comparison
    
    # If the median distance is 0, we need a fallback to avoid division by zero.
    sigma = median_dist.item() / np.sqrt(2.0) if median_dist > 0 else 1.0

    # Perform the first MMD comparison: Denoised vs Noisy 1
    # We compare the single denoised sample to the single noisy1 sample.
    mmd_sq_1 = mmd_loss(features_denoised_set, features_noisy_set1, sigma=sigma)
    
    # Perform the second MMD comparison: Denoised vs Noisy 2
    # We compare the single denoised sample to the single noisy2 sample.
    mmd_sq_2 = mmd_loss(features_denoised_set, features_noisy_set2, sigma=sigma)
    
    return mmd_sq_1.item(), mmd_sq_2.item()

def plot_denoising_results_from_npz(histograms_mmap, denoised_4d, method, plot_indices, directory="./plotsMMD", original_4d_shape=None, plotInDB=False, hu_values=None, initial_energies=None):
    """Generates plots comparing noisy, denoised, and true histograms from the saved data."""
    print("\n[INFO] Generating denoising plots...")
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
        
        # Transform to dB
        if plotInDB:
            noisy_input_np = 10 * np.log10(noisy_input_np + 1e-12)
            true_clean_np = 10 * np.log10(true_clean_np + 1e-12)
            denoised_output_np = 10 * np.log10(denoised_output_np + 1e-12)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # New plot title with HU and initial energy values
        fig.suptitle(f'Material: {hu_values[m_idx_to_plot] - 1000 :.0f} HU, Initial Energy: {initial_energies[e_idx_to_plot]:.1f} MeV', fontsize=17)

        im1 = axes[0].imshow(noisy_input_np.T, origin='lower', aspect='auto', cmap='Reds', extent=extent)
        axes[0].set_title('Noisy Input')
        plt.colorbar(im1, ax=axes[0], label='Probability (dB)')
        if method == 'normalization':
            axes[0].set_xlabel('Normalized Angle (a.u.)')
            axes[0].set_ylabel('Normalized Energy (a.u.)')
        else:
            axes[0].set_xlabel(r'$Angle\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
            axes[0].set_ylabel(r'$\frac{ln((E_i-E_f)/E_i)}{\sqrt{E_i}}$ (MeV$^{-1/2}$)')
        im2 = axes[1].imshow(denoised_output_np.T, origin='lower', aspect='auto', cmap='Reds', extent=extent)
        axes[1].set_title('Predicted Denoised')
        plt.colorbar(im2, ax=axes[1], label='Probability (dB)')
        if method == 'normalization':
            axes[1].set_xlabel('Normalized Angle (a.u.)')
            axes[1].set_ylabel('Normalized Energy (a.u.)')
        else:
            axes[1].set_xlabel(r'$Angle\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
            axes[1].set_ylabel(r'$\frac{ln((E_i-E_f)/E_i)}{\sqrt{E_i}}$ (MeV$^{-1/2}$)')
        im3 = axes[2].imshow(true_clean_np.T, origin='lower', aspect='auto', cmap='Reds', extent=extent)
        axes[2].set_title('Reference Noisy')
        plt.colorbar(im3, ax=axes[2], label='Probability (dB)')
        if method == 'normalization':
            axes[2].set_xlabel('Normalized Angle (a.u.)')
            axes[2].set_ylabel('Normalized Energy (a.u.)')
        else:
            axes[2].set_xlabel(r'$Angle\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
            axes[2].set_ylabel(r'$\frac{ln((E_i-E_f)/E_i)}{\sqrt{E_i}}$ (MeV$^{-1/2}$)')
        plt.tight_layout()
        plt.savefig(f'{directory}/denoising_example_full_run_M{m_idx_to_plot}_E{e_idx_to_plot}.pdf')
        plt.close(fig)
        
    print(f"[INFO] Generated {len(plot_indices)} denoising plots.")

def run_per_me_analysis(noisy_data, denoised_data):
    """
    Runs the MMD comparison for each individual (M, E) pair and aggregates the results.

    Args:
        noisy_data (np.ndarray): Memory-mapped data from the noisy file.
        denoised_data (np.ndarray): Memory-mapped data from the denoised file.
    
    Returns:
        dict: A dictionary with 'mmd1_results' and 'mmd2_results' arrays, and
              summary statistics.
    """
    M = denoised_data.shape[0]
    E = denoised_data.shape[1]
    
    mmd1_results = []
    mmd2_results = []
    
    print(f"[INFO] Running MMD comparison for all ({M} x {E}) = {M*E} pairs...")
    
    # Iterate through each unique (M, E) pair
    with tqdm(total=M*E, desc="Processing pairs") as pbar:
        for m_idx in range(M):
            for e_idx in range(E):
                # Extract the histograms for this specific pair
                denoised_hist = denoised_data[m_idx, e_idx]
                noisy1_hist = noisy_data[0, m_idx, e_idx]
                noisy2_hist = noisy_data[1, m_idx, e_idx]
                
                mmd1, mmd2 = calculate_mmd_per_me(denoised_hist, noisy1_hist, noisy2_hist)
                
                mmd1_results.append(mmd1)
                mmd2_results.append(mmd2)
                pbar.update(1)
    
    mmd1_results = np.array(mmd1_results)
    mmd2_results = np.array(mmd2_results)
    
    print("\n[ANALYSIS RESULTS]")
    print(f"   MMD^2 (Denoised vs Noisy 1) - Overall Stats:")
    print(f"\tAverage: {np.mean(mmd1_results):.6f}")
    print(f"\tStandard Deviation: {np.std(mmd1_results):.6f}")
    print(f"\tMin: {np.min(mmd1_results):.6f}")
    print(f"\tMax: {np.max(mmd1_results):.6f}")
    
    print(f"\n   MMD^2 (Denoised vs Noisy 2) - Overall Stats:")
    print(f"\tAverage: {np.mean(mmd2_results):.6f}")
    print(f"\tStandard Deviation: {np.std(mmd2_results):.6f}")
    print(f"\tMin: {np.min(mmd2_results):.6f}")
    print(f"\tMax: {np.max(mmd2_results):.6f}")

    return {
        'mmd1_results': mmd1_results,
        'mmd2_results': mmd2_results,
    }
    
def plot_mmd_histograms(mmd1_results, mmd2_results, directory="./plotsMMD"):
    """
    Generates and saves histograms of the MMD^2 results.

    Args:
        mmd1_results (np.ndarray): Array of MMD^2 values for Denoised vs Noisy 1.
        mmd2_results (np.ndarray): Array of MMD^2 values for Denoised vs Noisy 2.
        directory (str): The directory to save the plots.
    """
    print("\n[INFO] Generating MMD results plot...")

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Determine a reasonable number of bins for the histogram
    num_bins = 50

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.hist(mmd1_results, bins=num_bins, alpha=0.6, label='MMD² (Denoised vs Noisy 1)', color='skyblue')
    ax.hist(mmd2_results, bins=num_bins, alpha=0.6, label='MMD² (Denoised vs Noisy 2)', color='salmon')

    ax.set_title('Distribution of MMD² Scores')
    ax.set_xlabel('MMD² Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f'{directory}/mmd_distribution_histogram.pdf')
    plt.close(fig)

    print(f"[INFO] Generated MMD distribution plot saved to {directory}/mmd_distribution_histogram.pdf")

# --- Main Workflow ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a denoising CNN with real data from an NPZ file.")
    data_source_group = parser.add_mutually_exclusive_group(required=True)
    data_source_group.add_argument('--transformation', action='store_true', help="Use transformation method")
    data_source_group.add_argument('--normalization', action='store_true', help="Use normalization method")
    parser.add_argument("--npz", type=str, default=None, help="Optional path to a custom NPZ file.")
    args = parser.parse_args()
    
    # Data Loading and Initial Setup
    method = 'transformation' if args.transformation else 'normalization'
    noisy_path = args.npz if args.npz else (
        '../DenoisingDataTransSheet.npz' if method == 'transformation' else '../DenoisingDataNormSheet.npz'
    )
    denoised_path = f'../denoised_output_advanced_{method}.npz'
    
    # --- Main Workflow ---
    threshold = 1e-8
    plotInDB = True
    
    # Step 1: Load noisy data, the original denoised data, and the HU/energy values
    noisy, denoised, hu_values, initial_energies = load_data(noisy_path, denoised_path)
    
    if noisy is not None and denoised is not None and hu_values is not None and initial_energies is not None:
        print(f'[INFO] Threshold value: {threshold}')
        print(f'[INFO] Plot in DB: {plotInDB}')
    
        # Get shape info before cleaning
        original_4d_shape = denoised.shape
        num_materials = original_4d_shape[0]
        num_initial_energies = original_4d_shape[1]
    
        # Step 2: Apply cleaning and re-normalization to the data in memory
        print("[INFO] Applying threshold and keeping largest connected component in memory...")
        for m_idx in tqdm(range(num_materials), desc="Cleaning in memory"):
            for e_idx in range(num_initial_energies):
                histogram_2d = denoised[m_idx, e_idx, :, :]
                
                # Apply the two-step process in memory
                cleaned_histogram = keep_largest_component_from_threshold(
                    img=histogram_2d,
                    threshold=threshold,
                    fill_value=0
                )
                current_sum = np.sum(cleaned_histogram)
                if current_sum > 0.0001:
                    denoised[m_idx, e_idx, :, :] = cleaned_histogram / current_sum
                else:
                    denoised[m_idx, e_idx, :, :] = 0
        
        # Step 3: Run analysis and plot using the cleaned, in-memory data
        
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
            histograms_mmap=noisy, 
            denoised_4d=denoised, 
            method=method, 
            plot_indices=plot_indices, 
            directory=f'./plotsMMD_{method}_{threshold}',
            original_4d_shape=original_4d_shape, 
            plotInDB=plotInDB,
            hu_values=hu_values,
            initial_energies=initial_energies
        )
    
        # Run MMD analysis on the cleaned data
        results = run_per_me_analysis(noisy, denoised)
        plot_mmd_histograms(
            results['mmd1_results'], 
            results['mmd2_results'], 
            directory=f'./plotsMMD_{method}_{threshold}'
        )
        
        # Step 4: Save the final, cleaned data to a new NPZ file
        cleaned_denoised_path = f'./denoised_output_advanced_mask_{method}_{threshold}.npz'
        try:
            with np.load(denoised_path) as original_npz:
                data_to_save = {}
                data_to_save['probTable'] = denoised
                for key in original_npz.files:
                    if key != 'probTable':
                        data_to_save[key] = original_npz[key]
                np.savez_compressed(cleaned_denoised_path, **data_to_save)
                print(f"[INFO] Final cleaned data saved to '{cleaned_denoised_path}'.")
        except Exception as e:
            print(f"Error: Could not save the final cleaned data. {e}")
            
        print("\n[INFO] Analysis and saving completed!")