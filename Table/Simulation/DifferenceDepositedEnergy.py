import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os

# Matplotlib params
params = {
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,      
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'legend.fontsize': 14
}
pylab.rcParams.update(params)  # Apply changes

def readData(fileNameTOPAS, fileNameSimulation):
    
    dataTOPAS = np.load(fileNameTOPAS)
    dataSimulationTrans = np.load(fileNameSimulation)
    
    return dataTOPAS - dataSimulationTrans

def createPhysicalSpace(bigVoxel, voxelShapeBins, dt=1 / 3):
    # Create a range of coordinates from -bigVoxel to +bigVoxel with 200 bins
    xRange = np.linspace(-bigVoxel[0] / dt, bigVoxel[0] / dt, voxelShapeBins[0]) 
    yRange = np.linspace(-bigVoxel[1] / dt, bigVoxel[1] / dt, voxelShapeBins[1])
    zRange = np.linspace(-bigVoxel[2] / dt, bigVoxel[2] / dt, voxelShapeBins[2])
    
    return xRange, yRange, zRange

if __name__ == "__main__":
    # Define the file names
    fileNameTOPAS = "./projectionXZTOPAS.npy"
    numpyPath = "./Numpy/"
    numpyPathNorm = "./NumpyNormalized/"
    savePath = "./Plots/"
    fileNameSimulationTrans = f'{numpyPath}projectionXZSimulation_transformation.npy'
    fileNameSimulationNorm = f'{numpyPathNorm}projectionXZSimulation_normalization.npy'
    
    os.makedirs(savePath, exist_ok=True)
    
    voxelBig = (33, 33, 50)
    dt = 1 / 3
    voxelShapeBins = (50, 50, 300)

    dataTOPAS = np.load(fileNameTOPAS)
    dataSimulationTrans = np.load(fileNameSimulationTrans)
    dataSimulationNorm = np.load(fileNameSimulationNorm)
    differenceDataTrans = dataTOPAS - dataSimulationTrans
    differenceDataNorm = dataTOPAS - dataSimulationNorm

    # Create coordinate ranges
    xRange, _, zRange = createPhysicalSpace(voxelBig, voxelShapeBins)
    projectionXZTrans = np.sum(differenceDataTrans, axis=1)
    projectionXZNorm = np.sum(differenceDataNorm, axis=1)
    
    # Plot the projection
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    im1 = ax1.imshow(
        projectionXZTrans.T,
        extent=[xRange[0], xRange[-1], zRange[0], zRange[-1]],
        origin='lower',
        aspect='auto',
        cmap='Blues'
    )

    ax1.axhline(y=110, color='red', linestyle='--', linewidth=1.5, label='Z = 100 mm')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Z (mm)')
    ax1.set_title('Transform Variable Difference')

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Summed Energy Deposit (MeV)', fontsize=12)
    cbar1.ax.tick_params(labelsize=10)

    # Set limits
    ax1.set_xlim(-voxelBig[0] / dt, voxelBig[0] / dt)
    ax1.set_ylim(-voxelBig[2] / dt, voxelBig[2] / dt)

    # Save the first figure
    plt.tight_layout()
    plt.savefig(f"{savePath}DepositedEnergyDifferenceTransform.pdf", dpi=300)
    plt.close(fig1)

    # Second figure for Normalized Variable Difference
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im2 = ax2.imshow(
        projectionXZNorm.T,
        extent=[xRange[0], xRange[-1], zRange[0], zRange[-1]],
        origin='lower',
        aspect='auto',
        cmap='Blues'
    )

    ax2.axhline(y=110, color='red', linestyle='--', linewidth=1.5, label='Z = 100 mm')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Z (mm)')
    ax2.set_title('Normalized Variable Difference')

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Summed Energy Deposit (MeV)', fontsize=12)
    cbar2.ax.tick_params(labelsize=10)

    # Set limits
    ax2.set_xlim(-voxelBig[0] / dt, voxelBig[0] / dt)
    ax2.set_ylim(-voxelBig[2] / dt, voxelBig[2] / dt)

    # Save the second figure
    plt.tight_layout()
    plt.savefig(f"{savePath}DepositedEnergyDifferenceNormalized.pdf", dpi=300)
    plt.close(fig2)
            

    indxCenter = dataTOPAS.shape[0] // 2
    indxCenterSim = dataSimulationTrans.shape[0] // 2
    profileZTopas = dataTOPAS[indxCenter, :, :]
    profileZSimulationTrans = dataSimulationTrans[indxCenterSim, :, :]
    profileZSimulationNorm = dataSimulationNorm[indxCenterSim, :, :]
    profileZMeanTopas = profileZTopas.mean(axis=0)
    profileZMeanSimulationTrans = profileZSimulationTrans.mean(axis=0)
    profileZMeanSimulationNorm = profileZSimulationNorm.mean(axis=0)
                
    zIndex = 50
    profileXTopas = dataTOPAS[:, :, zIndex]
    profileXMeanTopas = profileXTopas.mean(axis=1)
    profileXSimulationTrans = dataSimulationTrans[:, :, zIndex]
    profileXSimulationNorm = dataSimulationNorm[:, :, zIndex]
    profileXMeanSimulationTrans = profileXSimulationTrans.mean(axis=1)
    profileXMeanSimulationNorm = profileXSimulationNorm.mean(axis=1)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(zRange, profileZMeanTopas, color='red', linestyle='-', linewidth=1, label="TOPAS")
    ax3.plot(zRange, profileZMeanSimulationTrans, color='blue', linestyle='--', linewidth=1, label="Variable Transform")
    ax3.plot(zRange, profileZMeanSimulationNorm, color='green', linestyle='--', linewidth=1, label="Variable Normalized")
    ax3.set_xlabel(r'Z voxel Index')
    # ax3.set_yscale('log')
    # ax3.set_ylim(1e4, 180000) 
    ax3.set_ylabel(r'Energy Deposit (MeV)')
    ax3.legend(loc='best', shadow=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f'{savePath}ProfileEnergyDepositZSimulation.pdf')
    plt.close(fig3)

    # Second figure for X voxel profile
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.plot(xRange, profileXMeanTopas, color='red', linestyle='-', linewidth=1, label="TOPAS")
    ax4.plot(xRange, profileXMeanSimulationTrans, color='blue', linestyle='--', linewidth=1, label="Variable Transform")
    ax4.plot(xRange, profileXMeanSimulationNorm, color='green', linestyle='--', linewidth=1, label="Variable Normalized")
    ax4.set_xlabel(r'X voxel Index')
    ax4.set_ylabel(r'Energy Deposit (MeV)')
    ax4.legend(loc='best', shadow=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f'{savePath}ProfileEnergyDepositXSimulation.pdf')
    plt.close(fig4)
