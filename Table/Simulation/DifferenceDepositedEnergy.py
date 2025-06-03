import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Matplotlib params
params = {
    'xtick.labelsize': 16,    
    'ytick.labelsize': 16,      
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'legend.fontsize': 12
}
pylab.rcParams.update(params)  # Apply changes

def readData(fileNameTOPAS, fileNameSimulation):
    
    dataTOPAS = np.load(fileNameTOPAS)
    dataSimulationTrans = np.load(fileNameSimulation)
    
    return dataTOPAS - dataSimulationTrans

def createPhysicalSpace(bigVoxel, voxelShapeBins):
    # Create a range of coordinates from -bigVoxel to +bigVoxel with 200 bins
    xRange = np.linspace(-bigVoxel[0], bigVoxel[0], voxelShapeBins[0]) 
    yRange = np.linspace(-bigVoxel[1], bigVoxel[1], voxelShapeBins[1])
    zRange = np.linspace(-bigVoxel[2], bigVoxel[2], voxelShapeBins[2])
    
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
    
    voxelBig = (100, 100, 150)
    voxelShapeBins = (50, 50, 100)
    # Create coordinate ranges
    xRange, yRange, zRange = createPhysicalSpace(voxelBig, voxelShapeBins)

    dataTOPAS = np.load(fileNameTOPAS)
    dataSimulationTrans = np.load(fileNameSimulationTrans)
    dataSimulationNorm = np.load(fileNameSimulationNorm)
    
    print(dataTOPAS.shape)
    print(dataSimulationTrans.shape)
    print(dataSimulationNorm.shape)
    
    # Compute differences and projections
    differenceDataTrans = dataTOPAS - dataSimulationTrans
    differenceDataNorm = dataTOPAS - dataSimulationNorm
    projectionXZTrans = np.sum(differenceDataTrans, axis=0)
    projectionXZNorm = np.sum(differenceDataNorm, axis=0)

    # Create a single figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # First subplot: Transform Difference
    im1 = ax1.imshow(
        projectionXZTrans.T,
        extent=[xRange[0], xRange[-1], zRange[0], zRange[-1]],
        origin='lower',
        aspect='auto',
        cmap='Blues'
    )
    # ax1.axhline(y=110, color='red', linestyle='--', linewidth=1.5, label='Z = 110 mm')
    ax1.set_xlabel('Y (mm)')
    ax1.set_ylabel('Z (mm)')
    ax1.set_title('Transform Variable Difference')
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
    cbar1.set_label('Summed Energy Deposit (MeV)', fontsize=15)
    cbar1.ax.tick_params(labelsize=10)
    ax1.set_xlim(-voxelBig[0], voxelBig[0])
    ax1.set_ylim(-voxelBig[2], voxelBig[2])

    # Second subplot: Normalized Difference
    im2 = ax2.imshow(
        projectionXZNorm.T,
        extent=[xRange[0], xRange[-1], zRange[0], zRange[-1]],
        origin='lower',
        aspect='auto',
        cmap='Blues'
    )
    # ax2.axhline(y=110, color='red', linestyle='--', linewidth=1.5, label='Z = 110 mm')
    ax2.set_xlabel('Y (mm)')
    ax2.set_title('Normalized Variable Difference')
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
    cbar2.set_label('Summed Energy Deposit (MeV)', fontsize=15)
    cbar2.ax.tick_params(labelsize=10)
    ax2.set_xlim(-voxelBig[0], voxelBig[0])
    ax2.set_ylim(-voxelBig[2], voxelBig[2])

    # Final layout and save
    plt.tight_layout()
    plt.savefig(f"{savePath}DepositedEnergyDifferenceYZ.pdf", dpi=300)
    plt.close(fig)
            

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
    
    profileYTopas = dataTOPAS[:, :, zIndex]
    profileYMeanTopas = profileYTopas.mean(axis=0)
    profileYSimulationTrans = dataSimulationTrans[:, :, zIndex]
    profileYSimulationNorm = dataSimulationNorm[:, :, zIndex]
    profileYMeanSimulationTrans = profileYSimulationTrans.mean(axis=0)
    profileYMeanSimulationNorm = profileYSimulationNorm.mean(axis=0)

    fig3, ax3 = plt.subplots(figsize=(8, 6))

    # Main plot: Energy deposits
    ax3.plot(zRange, profileZMeanTopas, color='red', linestyle='-', linewidth=1, label="TOPAS")
    ax3.plot(zRange, profileZMeanSimulationTrans, color='blue', linestyle='--', linewidth=1, label="Variable Transform")
    ax3.plot(zRange, profileZMeanSimulationNorm, color='green', linestyle='--', linewidth=1, label="Variable Normalized")
    ax3.set_xlabel(r'Z (mm)')
    ax3.set_ylabel(r'Summed Energy Deposit (MeV)')
    # ax3.set_title('Z Profile: Energy Deposit with Difference Inset')
    ax3.legend(loc='lower left', shadow=True, fancybox=True, framealpha=0.9)
    ax3.grid(True)

    # Create inset axes
    inset_ax3 = inset_axes(
        ax3,
        width="120%",         # Width relative to parent
        height="120%",        # Height relative to parent
        bbox_to_anchor=(0.175, 0.64, 0.3, 0.3),  # (x0, y0, width, height) in axes fraction (0 to 1)
        bbox_transform=ax3.transAxes,
        borderpad=0        
    )

    # Inset plot: Differences
    diffZTrans = profileZMeanTopas - profileZMeanSimulationTrans
    diffZNorm = profileZMeanTopas - profileZMeanSimulationNorm
    # diff = profileZMeanSimulationTrans - profileZMeanSimulationNorm
    # inset_ax3.plot(zRange, diff, color='blue', linestyle='-', linewidth=1, label='Transform - Normalized')
    inset_ax3.plot(zRange, diffZTrans, color='blue', linestyle='-', linewidth=1, label='TOPAS - Transform')
    inset_ax3.plot(zRange, diffZNorm, color='green', linestyle='-', linewidth=1, label='TOPAS - Normalized')
    inset_ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    # inset_ax3.set_title('Difference', fontsize=10)
    inset_ax3.set_xlabel(r'Z (mm)', fontsize=12)
    inset_ax3.set_ylabel(r'Difference (MeV)', fontsize=12)
    # inset_ax3.set_yscale('symlog')
    inset_ax3.tick_params(labelsize=8)
    inset_ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{savePath}ProfileEnergyDepositZSimulationBone.pdf')
    plt.close(fig3)


    fig4, ax4 = plt.subplots(figsize=(8, 6))

    # Main plot: Energy deposits
    ax4.plot(xRange, profileXMeanTopas, color='red', linestyle='-', linewidth=1, label="TOPAS")
    ax4.plot(xRange, profileXMeanSimulationTrans, color='blue', linestyle='--', linewidth=1, label="Variable Transform")
    ax4.plot(xRange, profileXMeanSimulationNorm, color='green', linestyle='--', linewidth=1, label="Variable Normalized")
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Summed Energy Deposit (MeV)')
    # ax4.set_title('X Profile: Energy Deposit with Difference Inset')
    ax4.legend(loc='best', shadow=True, fancybox=True, framealpha=0.9)
    ax4.grid(True)

    # Create inset axes: width, height, location (x0, y0, width, height) in axes coordinates
    inset_ax = inset_axes(
        ax4,
        width="105%",         
        height="105%",        
        bbox_to_anchor=(0.1, 0.6, 0.35, 0.35),  # (x0, y0, width, height) in axes fraction (0 to 1)
        bbox_transform=ax4.transAxes,
        borderpad=0        
    )

    # Inset plot: Differences
    diffXTrans = profileXMeanTopas - profileXMeanSimulationTrans
    diffXNorm = profileXMeanTopas - profileXMeanSimulationNorm
    #diff = profileXMeanSimulationTrans - profileXMeanSimulationNorm
    #inset_ax.plot(xRange, diff, color='blue', linestyle='-', linewidth=1, label='Transform - Normalized')
    inset_ax.plot(xRange, diffXTrans, color='blue', linestyle='-', linewidth=1, label='TOPAS - Transform')
    inset_ax.plot(xRange, diffXNorm, color='green', linestyle='-', linewidth=1, label='TOPAS - Normalized')
    inset_ax.axhline(0, color='black', linestyle='--', linewidth=1)
    # inset_ax.set_title('Difference', fontsize=10)
    inset_ax.set_xlabel(r'X (mm)', fontsize=12)
    inset_ax.set_ylabel(r'Difference (MeV)', fontsize=12)
    inset_ax.tick_params(labelsize=8)
    inset_ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{savePath}ProfileEnergyDepositXSimulationBone.pdf')
    plt.close(fig4)
    
    fig5, ax5 = plt.subplots(figsize=(8, 6))

    # Main plot: Energy deposits
    ax5.plot(xRange, profileYMeanTopas, color='red', linestyle='-', linewidth=1, label="TOPAS")
    ax5.plot(xRange, profileYMeanSimulationTrans, color='blue', linestyle='--', linewidth=1, label="Variable Transform")
    ax5.plot(xRange, profileYMeanSimulationNorm, color='green', linestyle='--', linewidth=1, label="Variable Normalized")
    ax5.set_xlabel('Y (mm)')
    ax5.set_ylabel('Summed Energy Deposit (MeV)')
    # ax5.set_title('X Profile: Energy Deposit with Difference Inset')
    ax5.legend(loc='best', shadow=True, fancybox=True, framealpha=0.9)
    ax5.grid(True)

    # Create inset axes: width, height, location (x0, y0, width, height) in axes coordinates
    inset_ax1 = inset_axes(
        ax5,
        width="105%",         
        height="105%",        
        bbox_to_anchor=(0.1, 0.6, 0.35, 0.35),  # (x0, y0, width, height) in axes fraction (0 to 1)
        bbox_transform=ax5.transAxes,
        borderpad=0           
    )

    # Inset plot: Differences
    diffYTrans = profileYMeanTopas - profileYMeanSimulationTrans
    diffYNorm = profileYMeanTopas - profileYMeanSimulationNorm
    #inset_ax1.plot(xRange, diff, color='blue', linestyle='-', linewidth=1, label='Transform - Normalized')
    inset_ax1.plot(xRange, diffYTrans, color='blue', linestyle='-', linewidth=1, label='TOPAS - Transform')
    inset_ax1.plot(xRange, diffYNorm, color='green', linestyle='-', linewidth=1, label='TOPAS - Normalized')
    inset_ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    # inset_ax1.set_title('Difference', fontsize=10)
    inset_ax1.set_xlabel(r'Y (mm)', fontsize=12)
    inset_ax1.set_ylabel(r'Difference (MeV)', fontsize=12)
    inset_ax1.tick_params(labelsize=8)
    inset_ax1.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{savePath}ProfileEnergyDepositYSimulationBone.pdf')
    plt.close(fig5)


    # Create 2D projections on the YZ plane (sum over X axis)
    projectionYZ_TOPAS = np.sum(dataTOPAS, axis=0)
    projectionYZ_Transform = np.sum(dataSimulationTrans, axis=0)
    projectionYZ_Normalized = np.sum(dataSimulationNorm, axis=0)

    # Plot the YZ projections
    figYZ, axesYZ = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    titles = ['TOPAS', 'Variable Transform', 'Variable Normalized']
    projections = [projectionYZ_TOPAS, projectionYZ_Transform, projectionYZ_Normalized]

    for ax, proj, title in zip(axesYZ, projections, titles):
        im = ax.imshow(
            proj.T, 
            extent=[yRange[0], yRange[-1], zRange[0], zRange[-1]],
            origin='lower',
            aspect='auto',
            cmap='Blues'
        )
        ax.set_title(title)
        ax.set_xlabel('Y (mm)')
        ax.set_ylabel('Z (mm)')
        figYZ.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"{savePath}ProjectionYZ.pdf", dpi=300)
    plt.close(figYZ)
    
    
    projectionXZ_TOPAS = np.sum(dataTOPAS, axis=1)
    projectionXZ_Transform = np.sum(dataSimulationTrans, axis=1)
    projectionXZ_Normalized = np.sum(dataSimulationNorm, axis=1)

    # Plot the XZ projections
    figXZ, axesXZ = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    projections = [projectionXZ_TOPAS, projectionXZ_Transform, projectionXZ_Normalized]

    for ax, proj, title in zip(axesXZ, projections, titles):
        im = ax.imshow(
            proj.T, 
            extent=[xRange[0], xRange[-1], zRange[0], zRange[-1]],
            origin='lower',
            aspect='auto',
            cmap='Blues'
        )
        ax.set_title(title)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        figXZ.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"{savePath}ProjectionXZ.pdf", dpi=300)
    plt.close(figXZ)