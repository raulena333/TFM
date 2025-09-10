import pandas as pd
import os
import numpy as np

def returnEnergyValueArray(filePath):
    """
    Reads a TOPAS CSV file and returns the voxel coordinates and values.
    
    Args:
        filePath (str): The path to the TOPAS CSV file.
        
    Returns:
        tuple: A tuple containing x, y, z, and value numpy arrays.
    """
    if not os.path.exists(filePath):
        print(f"Error: File {filePath} does not exist.")
        return None, None, None, None
    
    # Read the CSV file and assign column names
    try:
        data = pd.read_csv(filePath, header=None, comment='#', names=['x', 'y', 'z', 'value'])
    except Exception as e:
        print(f"Error reading file {filePath}: {e}")
        return None, None, None, None

    # Extract x, y, z, value as separate numpy arrays
    x = data['x'].values
    y = data['y'].values
    z = data['z'].values
    value = data['value'].values
    
    return x, y, z, value

def loadCombinedGrid(waterFilePath, mixtureFilePath, finalVoxelShape):
    """
    Loads data from two separate TOPAS CSV files (one for water, one for mixture),
    combines them, and returns a single numpy grid.
    
    Args:
        waterFilePath (str): Path to the CSV file for the water box.
        mixtureFilePath (str): Path to the CSV file for the mixture box.
        finalVoxelShape (tuple): The desired shape of the combined grid (e.g., (50, 50, 300)).
        
    Returns:
        np.array: The combined 3D numpy grid.
    """
    print(f"Loading data for water box from: {waterFilePath}")
    x_w, y_w, z_w, vals_w = returnEnergyValueArray(waterFilePath)
    
    print(f"Loading data for mixture box from: {mixtureFilePath}")
    x_m, y_m, z_m, vals_m = returnEnergyValueArray(mixtureFilePath)
    
    if x_w is None or x_m is None:
        return None

    # Create a single grid for the combined data
    combined_grid = np.zeros(finalVoxelShape)
    
    # Place the water data into the first half of the combined grid.
    # The z coordinates from the water CSV are already 0-149, which corresponds
    # to the first 150 slices of our new 300-slice grid.
    mask_w = vals_w > 0
    combined_grid[x_w[mask_w], y_w[mask_w], z_w[mask_w]] = vals_w[mask_w]

    # Place the mixture data into the second half of the combined grid.
    # We must add 150 to the z coordinate to place it after the water data.
    z_m_shifted = z_m + finalVoxelShape[2] // 2
    mask_m = vals_m > 0
    combined_grid[x_m[mask_m], y_m[mask_m], z_m_shifted[mask_m]] = vals_m[mask_m]
    
    # Flip the combined grid along the z-axis, as in the original code.
    combined_grid = combined_grid[:, :, ::-1]

    return combined_grid

if __name__ == "__main__":
    # The final combined grid shape will be 50x50x300
    finalVoxelShape = (50, 50, 300)

    # Define file paths for the two boxes for each quantity
    energyWaterFilePath = "./EnergyAtWaterByBinsTOPAS.csv"
    energyMixtureFilePath = "./EnergyAtMixtureByBinsTOPAS.csv"
    
    fluenceWaterFilePath = "./FluenceAtWaterByBinsTOPAS.csv"
    fluenceMixtureFilePath = "./FluenceAtMixtureByBinsTOPAS.csv"
    
    energyFluenceWaterFilePath = "./EnergyFluenceAtWaterByBinsTOPAS.csv"
    energyFluenceMixtureFilePath = "./EnergyFluenceAtMixtureByBinsTOPAS.csv"
    
    # Load and combine the 3D grids for each quantity
    energyGrid = loadCombinedGrid(energyWaterFilePath, energyMixtureFilePath, finalVoxelShape)
    fluenceGrid = loadCombinedGrid(fluenceWaterFilePath, fluenceMixtureFilePath, finalVoxelShape)
    energyFluenceGrid = loadCombinedGrid(energyFluenceWaterFilePath, energyFluenceMixtureFilePath, finalVoxelShape)

    if energyGrid is None or fluenceGrid is None or energyFluenceGrid is None:
        print("One or more grids failed to load. Exiting.")
    else:
        # Save the combined energyGrid
        np.save('energyDepositedTOPAS.npy', energyGrid)
        print("Combined energy grid saved to 'energyDepositedTOPAS.npy'")

        # Compute average energy: EnergyFluence / Fluence (with threshold)
        fluenceThreshold = 1e-10  # A more robust threshold to avoid division by zero
        meanEnergyGrid = np.zeros_like(energyGrid)
        mask = fluenceGrid > fluenceThreshold
        meanEnergyGrid[mask] = energyFluenceGrid[mask] / fluenceGrid[mask]

        # Save the average energy per voxel
        np.save('meanEnergyGridTOPAS.npy', meanEnergyGrid)
        print("Combined mean energy grid saved to 'meanEnergyGridTOPAS.npy'")
        
        # Total energy in grid
        totalEnergy = np.sum(energyGrid)
        print(f"Total energy: {totalEnergy:.6f} MeV")