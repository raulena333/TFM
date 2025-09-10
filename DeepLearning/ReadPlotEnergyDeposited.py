import pandas as pd
import os
import numpy as np

def returnEnergyValueArray(energyFile):
    if not os.path.exists(energyFile):
        print(f"Error: File {energyFile} does not exist.")
        return None
        
    # Read the CSV file and assign column names
    data = pd.read_csv(energyFile, header=None, comment='#', names=['x', 'y', 'z', 'energy'])

    # Extract x, y, z, energy as separate numpy arrays
    x = data['x'].values
    y = data['y'].values
    z = data['z'].values
    energy = data['energy'].values
        
    return x, y, z, energy

if __name__ == "__main__":
    
    mode = 'Hetero'

    voxelShapeBins = (50, 50, 300)
    energyFilePath = f"./EnergyAtPatientByBinsTOPAS{mode}.csv"
    fluencePath = f"./FluenceAtPatientByBinsTOPAS{mode}.csv"
    energyFluencePath = f"./EnergyFluenceAtPatientByBinsTOPAS{mode}.csv"
    
    zFlip = lambda z: voxelShapeBins[2] - 1 - z

    def loadGridFromCSV(filePath):
        x, y, z, vals = returnEnergyValueArray(filePath)
        mask = vals > 0
        x, y, z, vals = x[mask], y[mask], z[mask], vals[mask]
        z = zFlip(z)
        grid = np.zeros(voxelShapeBins)
        grid[x, y, z] = vals
        return grid  

    # Load the 3D grids
    energyGrid = loadGridFromCSV(energyFilePath)
    fluenceGrid = loadGridFromCSV(fluencePath)
    energyFluenceGrid = loadGridFromCSV(energyFluencePath)

    # Save energyGrid
    np.save('energyDepositedTOPAS.npy', energyGrid)

    # Compute average energy: EnergyFluence / Fluence (with threshold)
    fluenceThreshold = 1e0  # Threshold for fluence to avoid division by zero
    meanEnergyGrid = np.zeros_like(energyGrid)
    mask = fluenceGrid > fluenceThreshold
    meanEnergyGrid[mask] = energyFluenceGrid[mask] / fluenceGrid[mask]

    # Save the average energy per voxel
    np.save('meanEnergyGridTOPAS.npy', meanEnergyGrid)
    
    # Total energy in grid
    totalEnergy = np.sum(energyGrid)
    print(f"Total energy: {totalEnergy:.6f} MeV")

