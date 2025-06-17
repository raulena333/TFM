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

def createPhysicalSpace(bigVoxel, voxelShapeBins):
    # Create a range of coordinates from -bigVoxel to +bigVoxel with 200 bins
    xRange = np.linspace(-bigVoxel[0], bigVoxel[0], voxelShapeBins[0]) 
    yRange = np.linspace(-bigVoxel[1], bigVoxel[1], voxelShapeBins[1])
    zRange = np.linspace(-bigVoxel[2], bigVoxel[2], voxelShapeBins[2])
    
    return xRange, yRange, zRange

def get_voxel_index_z(z, z_min=-150.0, z_max=150.0, num_bins=300):
    voxel_size = (z_max - z_min) / num_bins  # = 1.0 mm
    index = int((z - z_min) // voxel_size)
    return index

# Define a helper to extract lateral profiles
def getLateralProfile(data, z_index):
    return data[:, :, z_index]


fileNameTOPAS = "./projectionXZTOPAS.npy"
numpyPath = "./Numpy/"
numpyPathNorm = "./NumpyNormalized/"
savePath = "./PlotsBetter/"
fileNameSimulationTrans = f'{numpyPath}projectionXZSimulation_transformation.npy'
fileNameSimulationNorm = f'{numpyPathNorm}projectionXZSimulation_normalization.npy'
    
os.makedirs(savePath, exist_ok=True)
    
voxelBig = (100, 100, 150)
voxelShapeBins = (50, 50, 300)
# Create coordinate ranges
xRange, yRange, zRange = createPhysicalSpace(voxelBig, voxelShapeBins)

dataSimulationTrans = np.load(fileNameSimulationTrans)
dataSimulationNorm = np.load(fileNameSimulationNorm)

# Load data from three simulations
sim1 = np.load(fileNameTOPAS)  # shape (Nx, Ny, Nz)
sim2 = np.load(fileNameSimulationTrans)
sim3 = np.load(fileNameSimulationNorm)

# Sum over x and y axes to get energy deposition along z-axis
sim1Z = np.sum(sim1, axis=(0, 1))
sim2Z = np.sum(sim2, axis=(0, 1))
sim3Z = np.sum(sim3, axis=(0, 1))

# Plot energy deposition vs z-depth
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(zRange, sim1Z, label='TOPAS', color='red', linestyle='-', linewidth=1)
plt.plot(zRange, sim2Z, label='Transformation', color='blue', linestyle='--', linewidth=1)
plt.plot(zRange, sim3Z, label='Normalization', color='green', linestyle='--', linewidth=1)
plt.xlabel('Z (mm)')
plt.ylabel('Energy deposited (MeV)')
plt.legend(loc='best', shadow=True, fancybox=True, framealpha=0.9)
plt.grid(True)

# Create inset axes for differences
ax_inset = inset_axes(
    ax,
    width="110%",
    height="110%",
    bbox_to_anchor=(0.14, 0.6, 0.3, 0.3),  # (x0, y0, width, height) in axes fraction (0 to 1)
    bbox_transform=ax.transAxes,
    borderpad=0        
)

diffSim2 = sim1Z - sim2Z
diffSim3 = sim1Z - sim3Z

ax_inset.plot(zRange, diffSim2, label='TOPAS - Transformation', color='blue', linestyle='-', linewidth=1)
ax_inset.plot(zRange, diffSim3, label='TOPAS - Normalization', color='green', linestyle='-', linewidth=1)
ax_inset.set_xlabel('Z (mm)', fontsize=10)
ax_inset.set_ylabel('Difference (MeV)', fontsize=10)
ax_inset.tick_params(labelsize=8)
ax_inset.grid(True)

plt.tight_layout()
plt.savefig(f'{savePath}EnergyDepositionZ.pdf')
plt.close()

# Calculate Bragg peak position for each simulation
braggIndex1 = np.argmax(sim1Z)

# Plot lateral profiles at different z-slices
zSlices = {
    'Entrance(z~-150)': get_voxel_index_z(-150),
    'Zone1(z~-50)': get_voxel_index_z(-50),
    'Zone2(z~0)': get_voxel_index_z(0),
    'Zone3(z~50)': get_voxel_index_z(50),
    'BraggPeak': braggIndex1
}

# Create inset axes for each z slice
for title, zIdx in zSlices.items():
    # X-profile
    fig, ax = plt.subplots(figsize=(8, 6))
    profiles_x = []
    for sim in [sim1, sim2, sim3]:
        profile = np.sum(getLateralProfile(sim, zIdx), axis=0)  # Sum over y for X-profile
        profiles_x.append(profile)

    for profile, label, color, linestyle in zip(profiles_x, ['TOPAS', 'Transformation', 'Normalization'], ['red', 'blue', 'green'], ['-', '--', '--']):
        ax.plot(xRange, profile, label=label, color=color, linestyle=linestyle, linewidth=1)

    ax.set_xlim(-voxelBig[0] / 2, voxelBig[0] / 2)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Energy (MeV)')
    ax.grid(True)
    ax.legend()
    
    # Inset for X-profile difference
    ax_inset = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=(0.12, 0.6, 0.3, 0.3),  # (x0, y0, width, height) in axes fraction (0 to 1)
        bbox_transform=ax.transAxes,
        borderpad=0  
    )
    
    diff_x_sim2 = profiles_x[0] - profiles_x[1]
    diff_x_sim3 = profiles_x[0] - profiles_x[2]
    
    ax_inset.plot(xRange, diff_x_sim2, label='TOPAS - Transformation', color='blue', linestyle='-')
    ax_inset.plot(xRange, diff_x_sim3, label='TOPAS - Normalization', color='green', linestyle='-')
    ax_inset.set_xlim(-voxelBig[0] / 2, voxelBig[0] / 2)
    ax_inset.set_xlabel('X (mm)', fontsize=10)
    ax_inset.set_ylabel('Difference (MeV)', fontsize=10)
    ax_inset.tick_params(labelsize=8)
    ax_inset.grid(True)

    plt.tight_layout()
    plt.savefig(f'{savePath}ProfileX_{title}.pdf')
    plt.close()

    # Y-profile
    fig, ax = plt.subplots(figsize=(8, 6))
    profiles_y = []
    for sim in [sim1, sim2, sim3]:
        profile = np.sum(getLateralProfile(sim, zIdx), axis=1)  # Sum over x for Y-profile
        profiles_y.append(profile)

    for profile, label, color, linestyle in zip(profiles_y, ['TOPAS', 'Transformation', 'Normalization'], ['red', 'blue', 'green'], ['-', '--', '--']):
        ax.plot(yRange, profile, label=label, color=color, linestyle=linestyle, linewidth=1)

    ax.set_xlim(-voxelBig[1] / 2, voxelBig[1] / 2)
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('Energy (MeV)')
    ax.grid(True)
    ax.legend()

    # Inset for Y-profile difference
    ax_inset = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=(0.12, 0.6, 0.3, 0.3),  # (x0, y0, width, height) in axes fraction (0 to 1)
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    diff_y_sim2 = profiles_y[0] - profiles_y[1]
    diff_y_sim3 = profiles_y[0] - profiles_y[2]

    ax_inset.plot(yRange, diff_y_sim2, label='TOPAS - Transformation', color='blue', linestyle='-')
    ax_inset.plot(yRange, diff_y_sim3, label='TOPAS - Normalization', color='green', linestyle='-')
    ax_inset.set_xlim(-voxelBig[1] / 2, voxelBig[1] / 2)
    ax_inset.set_xlabel('Y (mm)', fontsize=10)
    ax_inset.set_ylabel('Difference (MeV)', fontsize=10)
    ax_inset.tick_params(labelsize=8)
    ax_inset.grid(True)

    plt.tight_layout()
    plt.savefig(f'{savePath}ProfileY_{title}.pdf')
    plt.close()
    
    for title, zIdx in zSlices.items():
        fig, axs = plt.subplots(1, 4, figsize=(18, 5), sharex=True, sharey=True)
        fig.suptitle(f'XY Plane Energy Deposition at {title}', fontsize=16)

        sims = [sim1, sim2, sim3]
        labels = ['TOPAS', 'Transformation', 'Normalization']
        cmaps = ['Reds', 'Blues', 'Greens']

        for i, (sim, label, cmap) in enumerate(zip(sims, labels, cmaps)):
            xy_plane = getLateralProfile(sim, zIdx)  # shape: (Ny, Nx)
            im = axs[i].imshow(
                xy_plane.T,  # transpose to have X horizontal and Y vertical
                extent=[yRange[0], yRange[-1], xRange[0], xRange[-1]],
                origin='lower',
                cmap=cmap,
                aspect='equal'
            )
            axs[i].set_title(label)
            axs[i].set_xlabel('Y (mm)')
            if i == 0:
                axs[i].set_ylabel('X (mm)')
            fig.colorbar(im, ax=axs[i], shrink=0.75, label='Energy (MeV)')

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
        plt.savefig(f'{savePath}XYslice_{title}.pdf')
        plt.close()
