import numpy as np
import os
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import wasserstein_distance
from tqdm import tqdm

def calculate_centroid(data):
    """
    Calculates the centroid (mean) of a 1D or 2D dataset.
    
    Args:
        data (np.ndarray): The dataset.
        
    Returns:
        np.ndarray: The centroid of the dataset.
    """
    return np.mean(data, axis=0)

def calculate_mmd(data1, data2, gamma=1.0):
    """
    Calculates the Maximum Mean Discrepancy (MMD) using an RBF kernel.
    
    Args:
        data1 (np.ndarray): The first dataset.
        data2 (np.ndarray): The second dataset.
        gamma (float): The kernel parameter.
        
    Returns:
        float: The MMD value.
    """
    k_xx = rbf_kernel(data1, data1, gamma=gamma).mean()
    k_yy = rbf_kernel(data2, data2, gamma=gamma).mean()
    k_xy = rbf_kernel(data1, data2, gamma=gamma).mean()
    
    mmd_squared = k_xx + k_yy - 2 * k_xy
    return np.sqrt(max(0, mmd_squared))

if __name__ == "__main__":
    reference_file_name = '../../Plots/topasReferenceData.npz'
    sampled_file_name = 'sampleData.npz'

    # Check if both required files exist
    if not os.path.exists(reference_file_name):
        print(f"Error: The reference file '{reference_file_name}' was not found.")
        print("Please run the TOPAS processing script first to generate it.")
    elif not os.path.exists(sampled_file_name):
        print(f"Error: The sampled data file '{sampled_file_name}' was not found.")
        print("Please generate your sampled distribution and save it to this file.")
    else:
        # --- STEP 1: LOAD THE REFERENCE AND SAMPLED DATA ---
        print("Loading TOPAS reference data...")
        loaded_ref_data = np.load(reference_file_name)
        topas_energies = loaded_ref_data['energies']
        topas_angles = loaded_ref_data['angles']
        
        print("Loading sampled data...")
        loaded_sampled_data = np.load(sampled_file_name)
        sampled_energies = loaded_sampled_data['energies']
        sampled_angles = loaded_sampled_data['angles']
        
        # Combine energy and angle into single 2D arrays
        topas_distribution = np.vstack((topas_energies, topas_angles)).T
        sampled_distribution = np.vstack((sampled_energies, sampled_angles)).T

        # --- STEP 2: BATCH-BASED COMPARISON ---
        
        # Define batch parameters
        batch_size = 10_000  # Number of samples to use in each batch
        num_batches = 40     # Number of batches to compare
        
        # Lists to store metrics for each batch
        mmd_results = []
        wasserstein_energy_results = []
        wasserstein_angle_results = []

        print(f"\nComparing {num_batches} random batches of size {batch_size}...")

        for i in tqdm(range(num_batches), desc="Comparing batches"):
            # Randomly select indices for this batch
            topas_indices = np.random.choice(len(topas_distribution), batch_size, replace=False)
            sampled_indices = np.random.choice(len(sampled_distribution), batch_size, replace=False)

            topas_batch = topas_distribution[topas_indices, :]
            sampled_batch = sampled_distribution[sampled_indices, :]
            
            # Normalize the batch data for MMD
            topas_norm_batch = (topas_batch - np.mean(topas_batch, axis=0)) / np.std(topas_batch, axis=0)
            sampled_norm_batch = (sampled_batch - np.mean(sampled_batch, axis=0)) / np.std(sampled_batch, axis=0)

            # Calculate metrics for the current batch
            mmd_results.append(calculate_mmd(topas_norm_batch, sampled_norm_batch))
            wasserstein_energy_results.append(wasserstein_distance(topas_batch[:, 0], sampled_batch[:, 0]))
            wasserstein_angle_results.append(wasserstein_distance(topas_batch[:, 1], sampled_batch[:, 1]))

        # --- STEP 3: PRINT RESULTS ---
        print("\n--- Summary of Comparison Results ---")
        
        # Average Centroid Comparison
        topas_centroid = calculate_centroid(topas_distribution)
        sampled_centroid = calculate_centroid(sampled_distribution)
        print(f"\nAverage Centroid (Mean) of TOPAS energy: {topas_centroid[0]:.4f} MeV")
        print(f"Average Centroid (Mean) of Sampled energy: {sampled_centroid[0]:.4f} MeV")
        print(f"Energy Centroid Difference: {np.abs(topas_centroid[0] - sampled_centroid[0]):.4f} MeV")
        print("--------------------------------------------------")
        print(f"Average Centroid (Mean) of TOPAS angle: {topas_centroid[1]:.4f}°")
        print(f"Average Centroid (Mean) of Sampled angle: {sampled_centroid[1]:.4f}°")
        print(f"Angle Centroid Difference: {np.abs(topas_centroid[1] - sampled_centroid[1]):.4f}°")
        
        # MMD and Wasserstein results (averaged over batches)
        print(f"\nMMD (Maximum Mean Discrepancy) over batches: {np.mean(mmd_results):.4f} ± {np.std(mmd_results):.4f}")
        print(f"Wasserstein Distance for Energy over batches: {np.mean(wasserstein_energy_results):.4f} ± {np.std(wasserstein_energy_results):.4f}")
        print(f"Wasserstein Distance for Angle over batches: {np.mean(wasserstein_angle_results):.4f} ± {np.std(wasserstein_angle_results):.4f}")
