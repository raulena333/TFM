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

    voxelShapeBins = (50, 50, 300)
    energyFilePath = "./EnergyAtBoxByBinsTOPAS.csv"
    fluencePath = "./FluenceAtBoxByBinsTOPAS.csv"
    energyFluencePath = "./EnergyFluenceAtBoxByBinsTOPAS.csv"
    
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
    fluenceThreshold = 1e-6
    meanEnergyGrid = np.zeros_like(energyGrid)
    mask = fluenceGrid > fluenceThreshold
    meanEnergyGrid[mask] = energyFluenceGrid[mask] / fluenceGrid[mask]

    # Save the average energy per voxel
    np.save('meanEnergyGridTOPAS.npy', meanEnergyGrid)
















# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import matplotlib.pylab as pylab
# import os
# import matplotlib.colors as colors

# # Matplotlib params
# params = {
#     'xtick.labelsize': 14,    
#     'ytick.labelsize': 14,      
#     'axes.titlesize': 14,
#     'axes.labelsize': 14,
#     'legend.fontsize': 14
# }
# pylab.rcParams.update(params)  # Apply changes

# def returnEnergyValueArray(energyFile):
#     if not os.path.exists(energyFile):
#         print(f"Error: File {energyFile} does not exist.")
#         return None
    
#     # Read the CSV file and assign column names
#     data = pd.read_csv(energyFile, header=None, comment='#', names=['x', 'y', 'z', 'energy'])

#     # Extract x, y, z, energy as separate numpy arrays
#     x = data['x'].values
#     y = data['y'].values
#     z = data['z'].values
#     energy = data['energy'].values
    
#     return x, y, z, energy

# def createPhysicalSpace(bigVoxel, voxelShapeBins, dt=1 / 3):
#     # Create a range of coordinates from -bigVoxel to +bigVoxel with 200 bins
#     xRange = np.linspace(-bigVoxel[0] / dt, bigVoxel[0] / dt, voxelShapeBins[0]) 
#     yRange = np.linspace(-bigVoxel[1] / dt, bigVoxel[1] / dt, voxelShapeBins[1])
#     zRange = np.linspace(-bigVoxel[2] / dt, bigVoxel[2] / dt, voxelShapeBins[2])
    
#     return xRange, yRange, zRange

# if __name__ == "__main__":

#     savePath = "./PlotsTOPAS/"
#     energyFilePath = "./EnergyAtBoxByBinsTOPAS.csv"
#     voxelBig = (33, 33, 50)
#     dt = 1 / 3
#     voxelShapeBins = (50, 50, 300)

#     # Create directory if it doesn't exist
#     if not os.path.exists(savePath):
#         os.makedirs(savePath)
    
#     # Load the data
#     x, y, z, energies = returnEnergyValueArray(energyFilePath)

#     # Only keep non-zero energy deposits
#     mask = energies > 0
#     x = x[mask]
#     y = y[mask]
#     z = z[mask]
#     energies = energies[mask]
#     z = voxelShapeBins[2] - 1 - z
    
#     # Create an empty 3D grid
#     energyGrid = np.zeros(voxelShapeBins)

#     for xi, yi, zi, ei in zip(x, y, z, energies):
#         energyGrid[xi, yi, zi] = ei

#     projectionXZ = np.sum(energyGrid, axis=1)  # axis=1 is Y
#     np.save('projectionXZTOPAS.npy', energyGrid)

#     # Create coordinate ranges
#     xRange, _, zRange = createPhysicalSpace(voxelBig, voxelShapeBins)

#     # Plot the projection
#     fig, ax = plt.subplots(figsize=(8, 6))
#     im = ax.imshow(
#         projectionXZ.T,
#         extent=[xRange[0], xRange[-1], zRange[0], zRange[-1]],
#         origin='lower',
#         aspect='auto',
#         cmap='Blues'
#     )
    
#     # Draw horizontal line at Z = 100 mm
#     # ax.axhline(y=110, color='red', linestyle='--', linewidth=1.5, label='Z = 100 mm')

#     cbar = plt.colorbar(im, ax=ax)
#     cbar.set_label('Summed Energy Deposit (MeV)', fontsize=12)
#     cbar.ax.tick_params(labelsize=10)

#     ax.set_xlabel('X (mm)')
#     ax.set_ylabel('Z (mm)')
    
#     # Set limits
#     ax.set_xlim(-voxelBig[0] / dt, voxelBig[0] / dt)
#     ax.set_ylim(-voxelBig[2] / dt, voxelBig[2] / dt)

#     plt.tight_layout()
#     plt.savefig(f"{savePath}EnergyDeposit_XZ_ProjectionTOPAS.pdf", dpi=300)
#     plt.close(fig)
    
#     # Profile of the beam at X = 0 and X-Axis
#     indxCenter = energyGrid.shape[0] // 2
#     profileZ = energyGrid[indxCenter, :, :]
#     profileZMean = profileZ.mean(axis=0)
            
#     zIndex = 50
#     profileX = energyGrid[:, :, zIndex]
#     profileXMean = profileX.mean(axis=1)

#     fig1, ax1 =plt.subplots(1, 2, figsize=(11, 6))
#     ax1[0].plot(zRange, profileZMean)
#     ax1[0].set_xlabel(r'Z voxel Index')
#     ax1[0].set_ylabel(r'Energy Deposit (MeV)')
#     # ax1[0].set_xlim(- bigVoxelSize[2] / dt, + bigVoxelSize[2] / dt)
            
#     ax1[1].plot(xRange, profileXMean)
#     ax1[1].set_xlabel(r'X voxel Index')
#     ax1[1].set_ylabel(r'Energy Deposit (MeV)')
#     # ax1[1].set_xlim(-bigVoxelSize[0] / dt, + bigVoxelSize[0] / dt)
            
#     plt.tight_layout()
#     plt.savefig(f'{savePath}ProfilesEnergyDepositTOPAS.pdf')
#     plt.close(fig1)
    
    # # Convert voxel indices to physical coordinates
    # x_range, y_range, z_range = createPhysicalSpace(voxelBig, voxelShapeBins)
    # positionsX = x_range[x]
    # positionsY = y_range[y]
    # positionsZ = z_range[z]
    
    # # Create a new figure for the 3D scatter plot
    # fig3 = plt.figure(figsize=(10, 8))
    # ax = fig3.add_subplot(111, projection='3d')

    # # Scatter plot with color based on energy
    # sc = ax.scatter(positionsX, positionsZ, positionsY, c=energies, cmap='viridis', marker='o', s=1, alpha=0.8)

    # # Colorbar
    # cbar = plt.colorbar(sc, ax=ax)
    # cbar.set_label('Energy Deposit (MeV)', fontsize=12)
    # cbar.ax.tick_params(labelsize=10)

    # # Axis labels
    # ax.set_xlabel('X')
    # ax.set_ylabel('Z')
    # ax.set_zlabel('Y')
    
    # # Set limits
    # ax.set_xlim(-voxelBig / dt, voxelBig / dt)
    # ax.set_ylim(-voxelBig / dt, voxelBig / dt)
    # ax.set_zlim(-voxelBig / dt, voxelBig / dt)

    # # Save the figure
    # plt.savefig(f"{savePath}EnergyDeposit3D.pdf", dpi=300)
    # plt.close(fig3)
