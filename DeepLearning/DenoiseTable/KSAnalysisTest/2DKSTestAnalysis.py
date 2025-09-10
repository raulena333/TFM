import numpy as np
import argparse
from tqdm import tqdm
try:
    import KS2D
except ImportError:
    print("Error importing KS2D module")
    
def load_data(noisy_file_path, denoised_file_path):
    """
    Loads noisy and denoised data from .npz files and validates their shapes.

    Args:
        noisy_file_path (str): The file path to the noisy data .npz file.
                               Expected shape: [2, M, E, a, e]
        denoised_file_path (str): The file path to the denoised data .npz file.
                                  Expected shape: [M, E, a, e]
    
    Returns:
        tuple: A tuple containing the noisy data and denoised data, or (None, None)
               if an error occurs.
    """
    try:
        print("[INFO] Loading data...")
        # Load noisy data from .npz file
        with np.load(noisy_file_path, allow_pickle=False) as noisy_npz:
            noisy_data_shape = noisy_npz['histograms'].shape
            if noisy_data_shape[0] != 2:
                raise ValueError("Expected noisy data to be 5D with shape [2, M, E, a, e].")
        
        # Load denoised data from .npz file
        with np.load(denoised_file_path, allow_pickle=False) as denoised_npz:
            denoised_data_shape = denoised_npz['probTable'].shape
            if len(denoised_data_shape) != 4:
                raise ValueError("Expected denoised data to be 4D with shape [M, E, a, e].")

        print(f"[INFO] Data loaded successfully:")
        print(f"\t Noisy data shape: {noisy_data_shape}")
        print(f"\t Denoised data shape: {denoised_data_shape}")

        # Load data pickle
        noisy_data = np.load(noisy_file_path, mmap_mode='r')['histograms']
        denoised_data = np.load(denoised_file_path, mmap_mode='r')['probTable']
        
        return noisy_data, denoised_data
        
    except FileNotFoundError as e:
        print(f"Error: One or both files not found. Skipping analysis.")
        print(f"  {e}")
        return None, None
    except KeyError as e:
        print(f"Error: The 'histograms' key is missing from a .npz file. Skipping analysis.")
        print(f"  {e}")
        return None, None
    except ValueError as e:
        print(f"Error: Data has an incorrect shape. Skipping analysis.")
        print(f"  {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during data loading. Skipping analysis.")
        print(f"  {e}")
        return None, None

def probabilities_to_coords(prob_grid, num_samples):
    """
    Converts a 2D probability grid into a list of 2D coordinates using
    random sampling.

    This function treats the probability values as relative weights and
    samples a specified number of coordinates from the grid, which is a
    more robust method than the previous integer-counting approach.

    Args:
        prob_grid (np.ndarray): A 2D numpy array of probabilities, representing
                                a distribution.
        num_samples (int): The number of coordinate points to generate.

    Returns:
        np.ndarray: A 2D numpy array of shape [num_samples, 2] containing the
                    generated coordinates (x, y) for the statistical test.
    """
    # Normalize the probability grid to ensure it sums to 1
    total_prob = np.sum(prob_grid)
    if total_prob == 0:
        print("Warning: Input probability grid sums to zero. Returning empty coordinates.")
        return np.array([])
    normalized_probs = prob_grid / total_prob
    
    # Get the coordinates of each cell in the grid
    y_coords, x_coords = np.indices(normalized_probs.shape)
    
    # Flatten the coordinates and probabilities
    flattened_coords = np.vstack([x_coords.flatten(), y_coords.flatten()]).T
    flattened_probs = normalized_probs.flatten()

    # Sample from the flattened coordinates based on the flattened probabilities
    # We use np.random.choice with replacement to get a list of indices
    if np.sum(flattened_probs) == 0:
        return np.array([])
    
    chosen_indices = np.random.choice(
        len(flattened_probs), 
        size=num_samples, 
        p=flattened_probs,
        replace=True
    )
    
    # Use the chosen indices to get the final coordinate points
    coords = flattened_coords[chosen_indices]
    
    return coords

def compare_distributions(noisy_data, denoised_data):
    """
    Compares the denoised data to the two sets of noisy data using
    the 2D Kolmogorov-Smirnov (KS) test.

    This function iterates through the M and E dimensions, extracts the
    2D probability grids for a given pair, and uses the KS test to
    quantify the statistical difference between the distributions.

    Args:
        noisy_data (np.ndarray): The noisy data array of shape [2, M, E, a, e].
        denoised_data (np.ndarray): The denoised data array of shape [M, E, a, e].
    
    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary holds
               results for Denoised vs Noisy[0], and the second for
               Denoised vs Noisy[1]. Each dictionary has keys of the form (m, e)
               and values as a tuple (d, prob).
    """
    if noisy_data is None or denoised_data is None:
        print("Cannot compare distributions: data is missing.")
        return {}, {}

    # Extract dimensions for iteration
    num_m, num_e, _, _ = denoised_data.shape
    
    print("[INFO] Starting 2D KS test comparison...")
    
    # Dictionaries to store the results
    results_denoised_vs_noisy0 = {}
    results_denoised_vs_noisy1 = {}
    
    number_of_samples = 10_000

    # Iterate through M and E dimensions
    for m in tqdm(range(num_m), desc="Comparing M-values"):
        for e in range(num_e):
            # Extract the 2D probability grids
            denoised_grid = denoised_data[m, e, :, :]
            noisy_grid_0 = noisy_data[0, m, e, :, :]
            noisy_grid_1 = noisy_data[1, m, e, :, :]

            # Convert probability grids to coordinate arrays for the KS test
            denoised_coords = probabilities_to_coords(denoised_grid, number_of_samples)
            noisy_coords_0 = probabilities_to_coords(noisy_grid_0, number_of_samples)
            noisy_coords_1 = probabilities_to_coords(noisy_grid_1, number_of_samples)
            
            # Compare denoised data with noisy data set 0
            d0, prob0 = KS2D.ks2d2s(denoised_coords, noisy_coords_0)
            print('Hola')
            results_denoised_vs_noisy0[(m, e)] = (d0, prob0)

            # Compare denoised data with noisy data set 1
            d1, prob1 = KS2D.ks2d2s(denoised_coords, noisy_coords_1)
            results_denoised_vs_noisy1[(m, e)] = (d1, prob1)
            
    return results_denoised_vs_noisy0, results_denoised_vs_noisy1
            
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
    
    denoised_path = f'./denoised_output_advanced_{method}.npz'
    
    # Load the data using the provided function
    noisy_data, denoised_data = load_data(npz_path, denoised_path)
    
    # Run the comparison if data was loaded successfully
    results_n0, results_n1 = compare_distributions(noisy_data, denoised_data)

    # Print a summary of the results
    if results_n0:
        d0_values = [v[0] for v in results_n0.values()]
        prob0_values = [v[1] for v in results_n0.values()]
        avg_d0 = np.mean(d0_values)
        avg_prob0 = np.mean(prob0_values)
        print(f"\n--- Summary for Denoised vs Noisy[0] ---")
        print(f"Average KS statistic (d): {avg_d0:.4f}")
        print(f"Average p-value (prob): {avg_prob0:.4f}")

    if results_n1:
        d1_values = [v[0] for v in results_n1.values()]
        prob1_values = [v[1] for v in results_n1.values()]
        avg_d1 = np.mean(d1_values)
        avg_prob1 = np.mean(prob1_values)
        print(f"\n--- Summary for Denoised vs Noisy[1] ---")
        print(f"Average KS statistic (d): {avg_d1:.4f}")
        print(f"Average p-value (prob): {avg_prob1:.4f}")
