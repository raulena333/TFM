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

def createPhysicalSpace(bigVoxel, voxelShapeBins, dt=1 / 3):
    # Create a range of coordinates from -bigVoxel to +bigVoxel with 200 bins
    xRange = np.linspace(-bigVoxel[0] / dt, bigVoxel[0] / dt, voxelShapeBins[0]) 
    yRange = np.linspace(-bigVoxel[1] / dt, bigVoxel[1] / dt, voxelShapeBins[1])
    zRange = np.linspace(-bigVoxel[2] / dt, bigVoxel[2] / dt, voxelShapeBins[2])
    
    return xRange, yRange, zRange

if __name__ == "__main__":
    # Define the file names
    fileNameTOPAS = "./projectionXZTOPAS.npy"
    numpyPathNorm = "./NumpyNormalizedDeterministic/"
    # numpyPathNorm = "./NumpyNormalized/"
    savePath = "./Plots/"
    # linearInterpolation = f'{numpyPathNorm}projectionXZSimulation_normalizationLinear.npy'
    cubicInterpolation = f'{numpyPathNorm}projectionXZSimulation_normalization10.npy'  
    norm1 = f'{numpyPathNorm}projectionXZSimulation_normalization09.npy'
    norm2 = f'{numpyPathNorm}projectionXZSimulation_normalization11.npy'
    os.makedirs(savePath, exist_ok=True)
    
    voxelBig = (33, 33, 50)
    dt = 1 / 3
    voxelShapeBins = (50, 50, 300)

    dataTOPAS = np.load(fileNameTOPAS)
    # dataSimulationNorm1 = np.load(linearInterpolation)
    dataSimulationNorm = np.load(cubicInterpolation)
    dataSimulationNorm1 = np.load(norm1)
    dataSimulationNorm2 = np.load(norm2)

    # Create coordinate ranges
    xRange, _, zRange = createPhysicalSpace(voxelBig, voxelShapeBins)

    indxCenter = dataTOPAS.shape[0] // 2
    indxCenterSim = dataSimulationNorm1.shape[0] // 2
    profileZTopas = dataTOPAS[indxCenter, :, :]
    profileZSimulationNorm1 = dataSimulationNorm1[indxCenterSim, :, :]
    profileZSimulationNorm2 = dataSimulationNorm2[indxCenterSim, :, :]
    profileZSimulationNorm = dataSimulationNorm[indxCenter, :, :]
    
    profileZMeanTopas = profileZTopas.mean(axis=0)
    profileZMeanSimulationNorm1 = profileZSimulationNorm1.mean(axis=0)
    profileZMeanSimulationNorm2 = profileZSimulationNorm2.mean(axis=0)
    profileZMeanSimulationNorm = profileZSimulationNorm.mean(axis=0)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    # Main plot: Energy deposits
    ax3.plot(zRange, profileZMeanTopas, color='red', linestyle='-', linewidth=1, label="TOPAS")
    ax3.plot(zRange, profileZMeanSimulationNorm, color='blue', linestyle='--', linewidth=1, label='k=1.0')
    ax3.plot(zRange, profileZMeanSimulationNorm1, color='green', linestyle='--', linewidth=1, label="k=0.9")
    ax3.plot(zRange, profileZMeanSimulationNorm2, color='purple', linestyle='--', linewidth=1, label="k=1.1")
    ax3.set_xlabel(r'Z (mm)')
    ax3.set_ylabel(r'Summed Energy Deposit (MeV)')
    # ax3.set_title('Z Profile: Energy Deposit for Different Transformations')
    ax3.legend(
    loc='lower left',
    shadow=True,
    bbox_to_anchor=(0.32, 0.04),  # just x and y
    fancybox=True,
    framealpha=0.9
    )
    ax3.grid(True)

    # Create inset axes
    inset_ax3 = inset_axes(
        ax3,       
        width="120%",         # Width relative to parent
        height="120%",        # Height relative to parent
        bbox_to_anchor=(0.195, 0.59, 0.35, 0.35),  # (x0, y0, width, height) in axes fraction (0 to 1)
        bbox_transform=ax3.transAxes,
        borderpad=0        
    )

    # Inset plot: Differences
    diffZNorm1 = profileZMeanTopas - profileZMeanSimulationNorm1;
    diffZNorm2 = profileZMeanTopas - profileZMeanSimulationNorm2;
    diffZNorm = profileZMeanTopas - profileZMeanSimulationNorm;
    
    inset_ax3.plot(zRange, diffZNorm, color='blue', linestyle='-', linewidth=1, label='') 
    inset_ax3.plot(zRange, diffZNorm1, color='green', linestyle='-', linewidth=1, label='')
    inset_ax3.plot(zRange, diffZNorm2, color='purple', linestyle='-', linewidth=1, label='')
    inset_ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    # inset_ax3.set_title('Difference', fontsize=10)
    inset_ax3.set_xlabel(r'Z (mm)', fontsize=12)
    inset_ax3.set_ylabel(r'Difference (MeV)', fontsize=12)
    # inset_ax3.set_yscale('symlog')
    inset_ax3.tick_params(labelsize=8)
    inset_ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{savePath}ProfileEnergyDepositZSimulationWater.pdf')
    plt.close(fig3)
