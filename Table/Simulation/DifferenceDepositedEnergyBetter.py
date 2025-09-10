import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker

# Matplotlib params
params = {
    'xtick.labelsize': 18,    
    'ytick.labelsize': 18,      
    'axes.titlesize': 18,
    'axes.labelsize': 18,
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
    voxel_size = (z_max - z_min) / num_bins 
    index = int((z - z_min) // voxel_size)
    return index

def get_voxel_index_y(y, y_min=-100.0, y_max=100.0, num_bins=50):
    voxel_size = (y_max - y_min) / num_bins
    index = int((y - y_min) // voxel_size)
    return index

def get_voxel_index_x(x, x_min=-100.0, x_max=100.0, num_bins=50):
    voxel_size = (x_max - x_min) / num_bins
    index = int((x - x_min) // voxel_size)
    return index

# Define a helper to extract lateral profiles
def getLateralProfile(data, z_index):
    return data[:, :, z_index]

def getYZProfile(data, x_index):
    return data[x_index, :, :]

def getXZProfile(data, y_index):
    return data[:, y_index, :]

fileNameTOPAS = "./energyDepositedTOPAS.npy"

numpyPath = "./Numpy/"
numpyPathNorm = "./NumpyNormalized/"
fileNameSimulationTrans = f'{numpyPath}energyDepositedtransformation.npy'
fileNameSimulationNorm = f'{numpyPathNorm}energyDepositednormalization.npy'

savePath = "./PlotsBetter/" 
os.makedirs(savePath, exist_ok=True)

voxelBig = np.array((100., 100., 150.), dtype=np.float32)  # in mm
voxelShapeBins = np.array((50, 50, 300), dtype=np.int32)
voxelSize = 2 * voxelBig / voxelShapeBins # in mm
voxelVolume = voxelSize[0] * voxelSize[1] * voxelSize[2] # in mm^3
# Create coordinate ranges
xRange, yRange, zRange = createPhysicalSpace(voxelBig, voxelShapeBins)

dataSimulationTrans = np.load(fileNameSimulationTrans)
dataSimulationNorm = np.load(fileNameSimulationNorm)

# Load data from three simulations
sim1 = np.load(fileNameTOPAS)  # shape (Nx, Ny, Nz)
sim2 = np.load(fileNameSimulationTrans)
sim3 = np.load(fileNameSimulationNorm)
print(sim1.shape, sim2.shape, sim3.shape)

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
plt.ylabel('Energy deposition (MeV)')
plt.legend(loc='best', shadow=True, fancybox=True, framealpha=0.9)
plt.grid(True)

# Get the exponent of the main plot's y-axis scale
y_max_main = ax.get_ylim()[1]
main_exp = int(np.floor(np.log10(y_max_main)))

# Create inset axes for differences
ax_inset = inset_axes(
    ax,
    width="110%",
    height="110%",
    bbox_to_anchor=(0.18, 0.45, 0.3, 0.3),  # (x0, y0, width, height) in axes fraction (0 to 1)
    bbox_transform=ax.transAxes,
    borderpad=0        
)

diffSim2 = sim1Z - sim2Z
diffSim3 = sim1Z - sim3Z

ax_inset.plot(zRange, diffSim2, label='TOPAS - Transformation', color='blue', linestyle='-', linewidth=1)
ax_inset.plot(zRange, diffSim3, label='TOPAS - Normalization', color='green', linestyle='-', linewidth=1)

# Configure a ScalarFormatter to use the same scientific exponent as the main plot
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((main_exp, main_exp))
        
# Apply the formatter to the inset's y-axis
ax_inset.yaxis.set_major_formatter(formatter)

ax_inset.set_xlabel('Z (mm)', fontsize=13)
ax_inset.set_ylabel('Difference (MeV)', fontsize=13)
#ax_inset.set_title("Difference (MeV)", fontsize=10)
ax_inset.tick_params(labelsize=11)
ax_inset.grid(True)

plt.tight_layout()
plt.savefig(f'{savePath}EnergyDepositionZ.pdf')
plt.close()

# Integrate the curves to get the total deposited energy for each simulation
totalEnergyTOPAS = np.trapz(sim1Z, zRange)
totalEnergyTrans = np.trapz(sim2Z, zRange)
totalEnergyNorm = np.trapz(sim3Z, zRange)

print(f"Total energy deposited (TOPAS): {totalEnergyTOPAS} MeV")
print(f"Total energy deposited (Transformation): {totalEnergyTrans} MeV")
print(f"Total energy deposited (Normalization): {totalEnergyNorm} MeV")

# Calculate Bragg peak position for each simulation
braggIndex1 = np.argmax(sim1Z)

# Plot lateral profiles at different z-slices
zSlices = {
    'Entrance(z~--150)': get_voxel_index_z(-150),
    'Zone1(z~-100)': get_voxel_index_z(-100),
    'Zone2(z~-50)': get_voxel_index_z(-50),
    'Zone3(z~0)': get_voxel_index_z(0),
    'Zone4(z~50)': get_voxel_index_z(50),
    'Zone5(z~100)': get_voxel_index_z(100),
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
    ax.set_ylabel('Energy deposition (MeV)')
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
    
    # New code for inset formatting
    y_max_main_x = ax.get_ylim()[1]
    main_exp_x = int(np.floor(np.log10(y_max_main_x)))
    formatter_x = ticker.ScalarFormatter(useMathText=True)
    formatter_x.set_scientific(True)
    formatter_x.set_powerlimits((main_exp_x, main_exp_x))
    ax_inset.yaxis.set_major_formatter(formatter_x)

    ax_inset.set_xlim(-voxelBig[0] / 2, voxelBig[0] / 2)
    ax_inset.set_xlabel('X (mm)', fontsize=13)
    # ax_inset.set_ylabel('Difference (MeV)', fontsize=13)
    # ax_inset.set_title('Difference (MeV)', fontsize=10)
    ax_inset.tick_params(labelsize=11)
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
    
    # New code for inset formatting
    y_max_main_x = ax.get_ylim()[1]
    main_exp_x = int(np.floor(np.log10(y_max_main_x)))
    formatter_x = ticker.ScalarFormatter(useMathText=True)
    formatter_x.set_scientific(True)
    formatter_x.set_powerlimits((main_exp_x, main_exp_x))
    ax_inset.yaxis.set_major_formatter(formatter_x)
    
    ax_inset.set_xlim(-voxelBig[1] / 2, voxelBig[1] / 2)
    ax_inset.set_xlabel('Y (mm)', fontsize=13)
    # ax_inset.set_ylabel('Difference (MeV)', fontsize=10)
    # ax_inset.set_title('Difference (MeV)', fontsize=10)
    ax_inset.tick_params(labelsize=11)
    ax_inset.grid(True)

    plt.tight_layout()
    plt.savefig(f'{savePath}ProfileY_{title}.pdf')
    plt.close()
    
    for title, zIdx in zSlices.items():
        fig, axs = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
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


# Load saved mean energy grid
fluenceNameTOPAS = "./meanEnergyGridTOPAS.npy"
fluenceNameTrans = f'{numpyPath}meanEnergyGridtransformation.npy'
fluenceNameNorm = f'{numpyPathNorm}meanEnergyGridnormalization.npy'

meanEnergyGridTOPAS = np.load(fluenceNameTOPAS)  
meanEnergyGridTrans = np.load(fluenceNameTrans)  
meanEnergyGridNorm = np.load(fluenceNameNorm)  

# Compute profile
xIndex, yIndex = 25, 25
profileZMeanTOPAS = meanEnergyGridTOPAS[xIndex, yIndex, :]
profileZMeanTrans = meanEnergyGridTrans[xIndex, yIndex, :]
profileZMeanNorm = meanEnergyGridNorm[xIndex, yIndex, :]

# Compute mean profile over Y = 23 and 24 at fixed xIndex
# xIndex = 25
# yIndices = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]  # the Y-indices you want to average over
# yIndex='20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30'

# profileZMeanTOPAS = np.mean(meanEnergyGridTOPAS[xIndex, yIndices, :], axis=0)
# profileZMeanTrans = np.mean(meanEnergyGridTrans[xIndex, yIndices, :], axis=0)
# profileZMeanNorm = np.mean(meanEnergyGridNorm[xIndex, yIndices, :], axis=0)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(zRange, profileZMeanTOPAS, label='TOPAS', color='red', linestyle='-', linewidth=1)
plt.plot(zRange, profileZMeanTrans, label='Transformation', color='blue', linestyle='--', linewidth=1)
plt.plot(zRange, profileZMeanNorm, label='Normalization', color='green', linestyle='--', linewidth=1)
plt.xlabel('Z (mm)')
plt.ylabel('Mean Energy (MeV)')
#plt.xlim(-150, -100)
plt.legend(loc='best', shadow=True, fancybox=True, framealpha=0.9)
plt.grid(True)

# Create inset axes for differences
ax_inset = inset_axes(
    ax,
    width="110%",
    height="110%",
    bbox_to_anchor=(0.18, 0.2, 0.3, 0.3),  # (x0, y0, width, height) in axes fraction (0 to 1)
    bbox_transform=ax.transAxes,
    borderpad=0        
)

diffMeanTrans = profileZMeanTOPAS - profileZMeanTrans
diffMeanNorm = profileZMeanTOPAS - profileZMeanNorm

ax_inset.plot(zRange, diffMeanTrans, label='TOPAS - Transformation', color='blue', linestyle='-', linewidth=1)
ax_inset.plot(zRange, diffMeanNorm, label='TOPAS - Normalization', color='green', linestyle='-', linewidth=1)

ax_inset.set_xlabel('Z (mm)', fontsize=13)
#ax_inset.set_xlim(-150, -100)
ax_inset.set_ylabel('Difference (MeV)', fontsize=12)
ax_inset.tick_params(labelsize=11)
ax_inset.grid(True)

plt.tight_layout()
plt.savefig(f'{savePath}ProfileMeanEnergy_XIndex{xIndex}_YIndex{yIndex}.pdf')
plt.close()

# Save in .txt using 6 decimal places
saveTxt = np.column_stack((zRange, profileZMeanTOPAS, profileZMeanTrans, profileZMeanNorm))
np.savetxt(f'{savePath}ProfileMeanEnergy_XIndex{xIndex}_YIndex{yIndex}.txt', saveTxt, delimiter='\t', fmt='%.6f')

# Define x and y index sets
xIndices = [15, 20, 25, 30, 35]
yIndices = [15, 20, 25, 30, 35]

# Set up the figure and subplots
nRows, nCols = len(xIndices), len(yIndices)
fig, axs = plt.subplots(nRows, nCols, figsize=(3 * nCols, 2.5 * nRows), sharex=True, sharey=True)

for i, x in enumerate(xIndices):
    for j, y in enumerate(yIndices):
        ax = axs[i, j]

        # Get Z profiles at this (x, y)
        profileTOPAS = meanEnergyGridTOPAS[x, y, :]
        profileTrans = meanEnergyGridTrans[x, y, :]
        profileNorm = meanEnergyGridNorm[x, y, :]

        ax.plot(zRange, profileTOPAS, label='TOPAS', color='red', linewidth=1)
        ax.plot(zRange, profileTrans, label='Transf.', color='blue', linestyle='--', linewidth=1)
        ax.plot(zRange, profileNorm, label='Norm.', color='green', linestyle='--', linewidth=1)

        ax.set_title(f'X={x}, Y={y}', fontsize=8)
        ax.grid(True)
        ax.tick_params(labelsize=6)

        if i == nRows - 1:
            ax.set_xlabel("Z (mm)", fontsize=7)
        if j == 0:
            ax.set_ylabel("Mean Energy (MeV)", fontsize=7)

# Add a single legend at the top
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=9, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.suptitle('Mean Energy Profiles for All (X, Y) Index Pairs', fontsize=14)
plt.savefig(f'{savePath}All_XY_Profile_Grid.pdf')
plt.close()

# Extract the 2D XY slice (since Z has only 1 bin)
sliceIndex = 2
energy_trans = sim2[:, :, sliceIndex]
energy_norm = sim3[:, :, sliceIndex]
energy_topas = sim1[:, :, sliceIndex]

# Create meshgrid for plotting (center of voxel)
xCenters = xRange + (xRange[1] - xRange[0]) / 2
yCenters = yRange + (yRange[1] - yRange[0]) / 2
X, Y = np.meshgrid(xCenters, yCenters, indexing='ij')

# Plot transformation-based deposition
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

im0 = axs[0].pcolormesh(X, Y, energy_topas, cmap='viridis')
axs[0].set_title("TOPAS")
axs[0].set_xlabel("X [mm]")
axs[0].set_ylabel("Y [mm]")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].pcolormesh(X, Y, energy_trans, cmap='viridis')
axs[1].set_title("Transformation Sampling")
axs[1].set_xlabel("X [mm]")
axs[1].set_ylabel("Y [mm]")
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].pcolormesh(X, Y, energy_norm, cmap='viridis')
axs[2].set_title("Normalization Sampling")
axs[2].set_xlabel("X [mm]")
axs[2].set_ylabel("Y [mm]")
plt.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.savefig(os.path.join(savePath, "EnergyDepositionComparison.pdf"), dpi=300)
plt.close()

# Compute 1D profiles by summing over axes
profile_topas_x = np.sum(energy_topas, axis=1)  # sum over Y -> X profile
profile_topas_y = np.sum(energy_topas, axis=0)  # sum over X -> Y profile

profile_trans_x = np.sum(energy_trans, axis=1)
profile_trans_y = np.sum(energy_trans, axis=0)

profile_norm_x = np.sum(energy_norm, axis=1)
profile_norm_y = np.sum(energy_norm, axis=0)

# Plot X profiles
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(xCenters, profile_topas_x, label='TOPAS', lw=1, linestyle='-', color='red')
ax.plot(xCenters, profile_trans_x, label='Transformation', lw=1, linestyle='--', color='blue')
ax.plot(xCenters, profile_norm_x, label='Normalization', lw=1, linestyle='--', color='green')
ax.set_xlabel("X [mm]")
ax.set_ylabel("Energy Deposition [MeV]")
ax.set_title("Energy Deposition Profile along X")
ax.legend()
ax.grid(True)

# Inset axes for difference
ax_inset = inset_axes(
    ax,
    width="110%",
    height="110%",
    bbox_to_anchor=(0.1, 0.5, 0.3, 0.3),  # (x0, y0, width, height) in axes fraction (0 to 1)
    bbox_transform=ax.transAxes,
    borderpad=0        
)
diff_trans = profile_topas_x - profile_trans_x
diff_norm = profile_topas_x - profile_norm_x
ax_inset.plot(xCenters, diff_trans, label='TOPAS - Trans', color='blue', lw=1)
ax_inset.plot(xCenters, diff_norm, label='TOPAS - Norm', color='green', lw=1)
ax_inset.tick_params(labelsize=8)
ax_inset.set_title("Difference (MeV)", fontsize=10)
ax_inset.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(savePath, "EnergyProfile_X_withInset.pdf"), dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(yCenters, profile_topas_y, label='TOPAS', lw=1, linestyle='-', color='red')
ax.plot(yCenters, profile_trans_y, label='Transformation', lw=1, linestyle='--', color='blue')
ax.plot(yCenters, profile_norm_y, label='Normalization', lw=1, linestyle='--', color='green')
ax.set_xlabel("Y [mm]")
ax.set_ylabel("Energy Deposition [MeV]")
ax.set_title("Energy Deposition Profile along Y")
ax.legend()
ax.grid(True)

# Inset for difference
ax_inset = inset_axes(
    ax,
    width="110%",
    height="110%",
    bbox_to_anchor=(0.1, 0.5, 0.3, 0.3),  # (x0, y0, width, height) in axes fraction (0 to 1)
    bbox_transform=ax.transAxes,
    borderpad=0        
)
diff_trans = profile_topas_y - profile_trans_y
diff_norm = profile_topas_y - profile_norm_y
ax_inset.plot(yCenters, diff_trans, label='TOPAS - Trans', color='blue', lw=1)
ax_inset.plot(yCenters, diff_norm, label='TOPAS - Norm', color='green', lw=1)
ax_inset.set_title("Difference (MeV)", fontsize=10)
ax_inset.tick_params(labelsize=8)
ax_inset.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(savePath, "EnergyProfile_Y_withInset.pdf"), dpi=300)
plt.close()

xSlices = {
    'Zone1(x~-40)': get_voxel_index_x(-40),
    'Zone2(x~-20)': get_voxel_index_x(-20),
    'Zone3(x~-10)': get_voxel_index_x(-10),
    'Zone4(x~0)': get_voxel_index_x(0),
    'Zone5(x~10)': get_voxel_index_x(10),
    'Zone6(x~20)': get_voxel_index_x(20),
    'Zone7(x~40)': get_voxel_index_x(40),
}

savePath = "./PlotsBetter/ProfileYZ/"
os.makedirs(savePath, exist_ok=True)

# Plotting loop for YZ planes
for title, xIdx in xSlices.items():
    fig, axs = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
    fig.suptitle(f'YZ Plane Energy Deposition at {title}', fontsize=16)

    sims = [sim1, sim2, sim3]
    labels = ['TOPAS', 'Transformation', 'Normalization']
    cmaps = ['Reds', 'Blues', 'Greens']

    for i, (sim, label, cmap) in enumerate(zip(sims, labels, cmaps)):
        # Get the YZ plane slice
        yz_plane = getYZProfile(sim, xIdx) 

        im = axs[i].imshow(
            yz_plane,
            extent=[zRange[0], zRange[-1], yRange[0], yRange[-1]],
            origin='lower',
            cmap=cmap,
            aspect='equal'
        )
        axs[i].set_title(label)
        axs[i].set_xlabel('Z (mm)')
        if i == 0:
            axs[i].set_ylabel('Y (mm)')
        fig.colorbar(im, ax=axs[i], shrink=0.75, label='Energy (MeV)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{savePath}YZslice_{title}.pdf')
    plt.close()

ySlices = {
    'Zone1(y~-40)': get_voxel_index_y(-40),
    'Zone2(y~-20)': get_voxel_index_y(-20),
    'Zone3(y~-10)': get_voxel_index_y(-10),
    'Zone4(y~0)': get_voxel_index_y(0),
    'Zone5(y~10)': get_voxel_index_y(10),
    'Zone6(y~20)': get_voxel_index_y(20),
    'Zone7(y~40)': get_voxel_index_y(40),
}

savePath = "./PlotsBetter/ProfileXZ/"
os.makedirs(savePath, exist_ok=True)
  
for title, yIdx in ySlices.items():
    fig, axs = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
    fig.suptitle(f'XZ Plane Energy Deposition at {title}', fontsize=16)

    sims = [sim1, sim2, sim3]
    labels = ['TOPAS', 'Transformation', 'Normalization']
    cmaps = ['Reds', 'Blues', 'Greens']

    for i, (sim, label, cmap) in enumerate(zip(sims, labels, cmaps)):
        # Get the XZ plane slice
        xz_plane = getXZProfile(sim, yIdx)

        im = axs[i].imshow(
            xz_plane,
            extent=[zRange[0], zRange[-1], xRange[0], xRange[-1]],
            origin='lower',
            cmap=cmap,
            aspect='equal'
        )
        axs[i].set_title(label)
        axs[i].set_xlabel('Z (mm)')
        if i == 0:
            axs[i].set_ylabel('X (mm)')
        fig.colorbar(im, ax=axs[i], shrink=0.75, label='Energy (MeV)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{savePath}XZslice_{title}.pdf')
    plt.close()

#####################################################
import matplotlib.colors as colors

# --- Control Variable ---
useLog = True  # Set to False for a normal (linear) color scale

# --- Define the data and plot parameters once ---
sims = [sim1, sim2, sim3]
labels = ['TOPAS', 'Transformation', 'Normalization']
cmaps = ['Reds', 'Blues', 'Greens']

ySlices = {'Zone4(y~0)': get_voxel_index_y(0)}
xSlices = {'Zone4(x~0)': get_voxel_index_x(0)}

# Calculate Bragg peak position once
braggIndex1 = np.argmax(sim1Z)
bragg_z_pos = zRange[braggIndex1]

# --- Prepare data based on the 'useLog' flag ---
if useLog:
    # Collect all data planes to determine a single global vmin/vmax
    all_data_planes = []

    # Collect all XZ planes
    for sim in sims:
        for _, yIdx in ySlices.items():
            xz_plane = getXZProfile(sim, yIdx)
            # Handle zero values for log scale
            xz_plane = np.where(xz_plane > 0, xz_plane, 1e-12)
            all_data_planes.append(xz_plane)

    # Collect all YZ planes
    for sim in sims:
        for _, xIdx in xSlices.items():
            yz_plane = getYZProfile(sim, xIdx)
            # Handle zero values for log scale
            yz_plane = np.where(yz_plane > 0, yz_plane, 1e-12)
            all_data_planes.append(yz_plane)

    # Calculate the single global min and max
    global_min = min(plane.min() for plane in all_data_planes)
    global_max = max(plane.max() for plane in all_data_planes)
    
    # Define the normalization object
    norm = colors.LogNorm(vmin=global_min, vmax=global_max)
else:
    # No normalization object needed for linear scale
    norm = None

# --- Plotting XZ planes ---
savePath_xz = "./PlotsBetter/ProfileXZ/"
os.makedirs(savePath_xz, exist_ok=True)

for title, yIdx in ySlices.items():
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    
    for i, (sim, label, cmap) in enumerate(zip(sims, labels, cmaps)):
        xz_plane = getXZProfile(sim, yIdx)
        if useLog:
            xz_plane = np.where(xz_plane > 0, xz_plane, 1e-12)

        im = axs[i].imshow(
            xz_plane,
            extent=[zRange[0], zRange[-1], xRange[0], xRange[-1]],
            origin='lower',
            cmap=cmap,
            aspect='equal',
            norm=norm
        )
        axs[i].set_title(label)
        axs[i].set_xlabel('Z (mm)')
        if i == 0:
            axs[i].set_ylabel('X (mm)')
        fig.colorbar(im, ax=axs[i], shrink=0.75, label='Energy (MeV)')
        axs[i].axvline(bragg_z_pos, color='k', linestyle='--', linewidth=1.5)
        axs[i].axvline(0, color='k', linestyle=':', linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(f'{savePath_xz}XZslice_{title}.pdf')
    plt.close()

# --- Plotting YZ planes ---
savePath_yz = "./PlotsBetter/ProfileYZ/"
os.makedirs(savePath_yz, exist_ok=True)

for title, xIdx in xSlices.items():
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    for i, (sim, label, cmap) in enumerate(zip(sims, labels, cmaps)):
        yz_plane = getYZProfile(sim, xIdx)
        if useLog:
            yz_plane = np.where(yz_plane > 0, yz_plane, 1e-12)

        im = axs[i].imshow(
            yz_plane,
            extent=[zRange[0], zRange[-1], yRange[0], yRange[-1]],
            origin='lower',
            cmap=cmap,
            aspect='equal',
            norm=norm
        )
        axs[i].set_title(label)
        axs[i].set_xlabel('Z (mm)')
        if i == 0:
            axs[i].set_ylabel('Y (mm)')
        fig.colorbar(im, ax=axs[i], shrink=0.75, label='Energy Deposition (MeV)')
        axs[i].axvline(bragg_z_pos, color='k', linestyle='--', linewidth=1.5)
        axs[i].axvline(0, color='k', linestyle=':', linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(f'{savePath_yz}YZslice_{title}.pdf')
    plt.close()