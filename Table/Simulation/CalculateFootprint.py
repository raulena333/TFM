import numpy as np
import os
import sys

def convert_bytes_to_human_readable(size_in_bytes):
    """
    Converts a size in bytes to a human-readable string (e.g., KB, MB, GB).
    
    Args:
        size_in_bytes (int): The size in bytes.
    
    Returns:
        str: A string representing the size with an appropriate unit.
    """
    # Define the units for conversion
    units = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    size = float(size_in_bytes)
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"

def analyze_npz_file(file_path):
    """
    Loads a .npz file, calculates its size on disk, and estimates its memory usage.
    
    Args:
        file_path (str): The full path to the .npz file.
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_path}' was not found.")
            return

        # 1. Calculate and display the size on disk
        # This is the actual size of the compressed file on your hard drive.
        disk_size = os.path.getsize(file_path)
        print("--- File Size Analysis ---")
        print(f"File on disk: {convert_bytes_to_human_readable(disk_size)}")
        print("-" * 25)

        # 2. Load the .npz file and estimate its memory usage
        # The np.load() function returns an NpzFile object, which is a dictionary-like container.
        # It's important to be in a 'with' statement to ensure the file is closed properly.
        with np.load(file_path, allow_pickle=True) as data:
            total_memory_size = 0
            
            # Print a list of all arrays and their sizes within the .npz file
            print("Contents of .npz file:")
            for key in data.keys():
                # Access the numpy array for the current key
                array = data[key]
                
                # Calculate the size of a single array in bytes
                # array.size is the total number of elements in the array.
                # array.itemsize is the size of one element in bytes (e.g., 4 for a 32-bit integer).
                array_size = array.size * array.itemsize
                total_memory_size += array_size
                
                # Display the details for each array
                print(f"  - Array '{key}':")
                print(f"    - Shape: {array.shape}")
                print(f"    - Data Type: {array.dtype}")
                print(f"    - Estimated Memory Usage: {convert_bytes_to_human_readable(array_size)}")

            print("-" * 25)
            # Display the total estimated memory usage for all arrays combined
            print(f"Total estimated memory occupied: {convert_bytes_to_human_readable(total_memory_size)}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
if __name__ == "__main__":
    npz_file_path = '../Table/4DTableNormSheet.npz'
    
    analyze_npz_file(npz_file_path)