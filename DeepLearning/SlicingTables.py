import numpy as np
import os
import sys
import argparse

def process_and_save_npz(input_file_path: str, output_file_path: str):
    """
    Loads an NPZ file, extracts a slice of the 'histograms' array,
    and saves the modified data to a new NPZ file.

    Args:
        input_file_path (str): The path to the original .npz file.
        output_file_path (str): The path where the new .npz file will be saved.
    """
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at '{input_file_path}'", file=sys.stderr)
        return

    print(f"Loading data from: {input_file_path}")
    try:
        # Load the original data. Using mmap_mode can be slow for slicing,
        # so we load it fully into memory for this operation.
        with np.load(input_file_path, allow_pickle=True) as data:
            # Create a dictionary to hold the data for the new file
            new_data = {}

            # --- Process and slice the 'histograms' array ---
            if 'histograms' in data:
                print("Processing 'histograms' array...")
                original_histograms = data['histograms']
                
                # Check the shape to ensure it's safe to slice
                if original_histograms.ndim >= 1 and original_histograms.shape[0] > 0:
                    # Slice the array to get the first element along the first axis
                    sliced_histograms = original_histograms[0]
                    new_data['probTable'] = sliced_histograms
                    print(f" - Sliced 'histograms' array. New shape: {sliced_histograms.shape}")
                else:
                    print("Warning: 'histograms' array is empty or has an unexpected shape. Skipping slicing.")
                    # Optionally, you can decide to save the original or skip it entirely
                    new_data['histograms'] = original_histograms
            else:
                print("Warning: 'histograms' key not found in the file.")
            
            # --- Copy the other arrays ---
            print("Copying other arrays...")
            for key in data.files:
                if key != 'histograms':
                    new_data[key] = data[key]
                    print(f" - Copied key: '{key}', shape: {new_data[key].shape}")

            # --- Save the new data to a file ---
            np.savez(output_file_path, **new_data)
            print(f"\nSuccessfully saved the new NPZ file to: {output_file_path}")
            print(f"New file contains keys: {list(new_data.keys())}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

# --- Example Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save NPZ files based on a specified method.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--transformation', action='store_true', help="Use transformation method")
    group.add_argument('--normalization', action='store_true', help="Use normalization method")
    args = parser.parse_args()
    
    if args.transformation:
        method = 'transformation'
        input_file = './DenoisingDataTransSheet.npz'
        output_file = './DenoisingDataTransSheetSliced.npz'
    else: # This will be args.normalization
        method = 'normalization'
        input_file = './DenoisingDataNormSheet.npz'
        output_file = './DenoisingDataNormSheetSliced.npz'

    process_and_save_npz(input_file, output_file)

    # Optional: Verify the new file's contents
    print("\n--- Verifying the new file ---")
    if os.path.exists(output_file):
        try:
            with np.load(output_file, mmap_mode='r') as new_data:
                print(f"New file '{output_file}' contains {len(new_data.files)} arrays:")
                for key in new_data.files:
                    print(f" - Key: '{key}', Shape: {new_data[key].shape}, Size: {new_data[key].nbytes / 1024**2:.2f} MB")
        except Exception as e:
            print(f"Error verifying the new file: {e}")