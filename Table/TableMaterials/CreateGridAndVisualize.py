import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def createPhysicalSpace(bigVoxel, voxelShapeBins):
    # Create a range of coordinates from -bigVoxel to +bigVoxel with 200 bins
    xRange = np.linspace(-bigVoxel[0], bigVoxel[0], voxelShapeBins[0]) 
    yRange = np.linspace(-bigVoxel[1], bigVoxel[1], voxelShapeBins[1])
    zRange = np.linspace(-bigVoxel[2], bigVoxel[2], voxelShapeBins[2])
    
    return xRange, yRange, zRange

# Define voxel shape
voxelShapeBins = (50, 50, 300)
bigVoxelSize = np.array((100., 100., 150.), dtype=np.float32)  # Physical size in mm
xBins, yBins, zBins = voxelShapeBins

# Material IDs 
LUNG = 0
WATER = 1
BONE = 2
SOFT = 3

materialGrid = np.zeros(voxelShapeBins, dtype=np.int16)

# # Divide Z into 4 blocks
# Nz = voxelShapeBins[2]
# quarter = Nz // 4

# materialGrid[:, :, 0:quarter] = LUNG
# materialGrid[:, :, quarter:2*quarter] = WATER
# materialGrid[:, :, 2*quarter:3*quarter] = BONE
# materialGrid[:, :, 3*quarter:] = SOFT

# Z = 0–74:      LUNG
# Z = 75–149:    WATER
# Z = 150–224:   BONE
# Z = 225–299:   SOFT

# Save the material grid to a file for python and C++ usage
np.save("./materialGrid.npy", materialGrid)

# Flip the z-axis (axis=2 in your original grid)
materialGrid_flipped = materialGrid[:, :, ::-1]
materialGridTOPAS = np.transpose(materialGrid_flipped, (2, 0, 1))
materialGridTOPAS.flatten(order='C').astype(np.int16).tofile("materialGrid.dat")

xRange, yRange, zRange = createPhysicalSpace(bigVoxelSize, voxelShapeBins)
# Your colors and labels for the materials
colors = ['cyan', 'blue', 'brown', 'pink']
labels = ['Lung (0)', 'Water (1)', 'Bone (2)', 'Soft (3)']

# Create the colormap and norm as before
cmap = mcolors.ListedColormap(colors)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Axial slice (XY plane at middle Z)
zSlice = materialGrid[:, :, 110]
im0 = axs[0].imshow(zSlice, cmap=cmap, norm=norm)
im0 = axs[0].imshow(
    zSlice,
    cmap=cmap,
    norm=norm,
    extent=[xRange[0], xRange[-1], yRange[0], yRange[-1]],
    origin='lower'
)
axs[0].set_title(f'Axial Slice at Z = 110 mm')
axs[0].set_xlabel('X-axis')
axs[0].set_ylabel('Y-axis')

# Sagittal slice (YZ plane at middle X)
xSlice = materialGrid[xBins // 2, :, :]
im1 = axs[1].imshow(xSlice.T, cmap=cmap, norm=norm)
im1 = axs[1].imshow(
    xSlice.T, 
    cmap=cmap,
    norm=norm,
    extent=[yRange[0], yRange[-1], zRange[0], zRange[-1]],
    origin='lower'
)
axs[1].set_title(f'Sagittal Slice at X={xBins // 2} mm')
axs[1].set_xlabel('Y-axis')
axs[1].set_ylabel('Z-axis')

# Coronal slice (XZ plane at middle Y)
ySlice = materialGrid[:, yBins // 2, :]
im2 = axs[2].imshow(ySlice.T, cmap=cmap, norm=norm)  
im2 = axs[2].imshow(
    ySlice.T,
    cmap=cmap,
    norm=norm,
    extent=[xRange[0], xRange[-1], zRange[0], zRange[-1]],
    origin='lower'
)
axs[2].set_title(f'Coronal Slice at Y={yBins // 2} mm')
axs[2].set_xlabel('X-axis')
axs[2].set_ylabel('Z-axis')

plt.tight_layout()
plt.savefig("MaterialGridSlices.pdf")
plt.close()
