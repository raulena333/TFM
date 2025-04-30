import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import os
import re
import subprocess

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
    dataSimulation = np.load(fileNameSimulation)
    
    return dataTOPAS - dataSimulation

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
    savePath = "./Plots/"
    fileNameSimulation = f'{numpyPath}projectionXZSimulation.npy'
    
    os.makedirs(savePath, exist_ok=True)
    
    voxelBig = (33, 33, 50)
    dt = 1 / 3
    voxelShapeBins = (50, 50, 300)

    dataTOPAS = np.load(fileNameTOPAS)
    dataSimulation = np.load(fileNameSimulation)
    differenceData = dataTOPAS - dataSimulation

    # Create coordinate ranges
    xRange, _, zRange = createPhysicalSpace(voxelBig, voxelShapeBins)
    projectionXZ = np.sum(differenceData, axis=1)
    
    # Plot the projection
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        projectionXZ.T,
        extent=[xRange[0], xRange[-1], zRange[0], zRange[-1]],
        origin='lower',
        aspect='auto',
        cmap='Blues'
    )

    ax.axhline(y=110, color='red', linestyle='--', linewidth=1.5, label='Z = 100 mm')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Summed Energy Deposit (MeV)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')

    # Set limits
    ax.set_xlim(-voxelBig[0] / dt, voxelBig[0] / dt)
    ax.set_ylim(-voxelBig[2] / dt, voxelBig[2] / dt)

    plt.tight_layout()
    plt.savefig(f"{savePath}DepositedEnergyDifference.pdf", dpi=300)
    plt.close(fig)
            
    # Profile of the beam at X = 0 and X-Axis
    indxCenter = dataTOPAS.shape[0] // 2
    profileZTopas = dataTOPAS[indxCenter, :, :]
    profileZSimulation = dataSimulation[indxCenter, :, :]
    profileZMeanTopas = profileZTopas.mean(axis=0)
    profileZMeanSimulation = profileZSimulation.mean(axis=0)
                
    zIndex = 50
    profileXTopas = dataTOPAS[:, :, zIndex]
    profileXMeanTopas = profileXTopas.mean(axis=1)
    profileXSimulation = dataSimulation[:, :, zIndex]
    profileXMeanSimulation = profileXSimulation.mean(axis=1)

    fig1, ax1 =plt.subplots(1, 2, figsize=(10, 6))
    ax1[0].plot(zRange, profileZMeanTopas, color='red', linestyle='dashed', linewidth=1, label = "TOPAS")
    ax1[0].plot(zRange, profileZMeanSimulation, color='blue', linestyle='-', linewidth=1, label = "Simulation")
    ax1[0].set_xlabel(r'Z voxel Index')
    ax1[0].set_ylabel(r'Energy Deposit (MeV)')
    # ax1[0].set_yscale('log')
    # ax1[0].set_xlim(- bigVoxelSize[2] / dt, + bigVoxelSize[2] / dt)
                
    ax1[1].plot(xRange, profileXMeanTopas, color='red', linestyle='dashed', linewidth=1, label = "TOPAS")
    ax1[1].plot(xRange, profileXMeanSimulation, color='blue', linestyle='-', linewidth=1, label = "Simulation")
    ax1[1].set_xlabel(r'X voxel Index')
    ax1[1].set_ylabel(r'Energy Deposit (MeV)')
    # ax1[1].set_yscale('log')
    # ax1[1].set_xlim(-bigVoxelSize[0] / dt, + bigVoxelSize[0] / dt)
                
    plt.tight_layout()
    plt.savefig(f'{savePath}ProfilesEnergyDepositSimulation.pdf')
    plt.close(fig1)
