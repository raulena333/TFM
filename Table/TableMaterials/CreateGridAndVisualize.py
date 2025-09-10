import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.pylab as pylab

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
    xRange = np.linspace(-bigVoxel[0], bigVoxel[0], voxelShapeBins[0])
    yRange = np.linspace(-bigVoxel[1], bigVoxel[1], voxelShapeBins[1])
    zRange = np.linspace(-bigVoxel[2], bigVoxel[2], voxelShapeBins[2])
    return xRange, yRange, zRange

# Define voxel shape and physical size
voxelShapeBins = (50, 50, 300)
bigVoxelSize = np.array((100., 100., 150.), dtype=np.float32)
xBins, yBins, zBins = voxelShapeBins

# Material IDs
LUNG = 0
WATER = 1
BONE = 2
SOFT = 3

# Initialize the grid with LUNG tissue
materialGrid = np.zeros(voxelShapeBins, dtype=np.int16)

# Get the physical space coordinates
x, y, z = createPhysicalSpace(bigVoxelSize, voxelShapeBins)

# Create a meshgrid for vectorized operations
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# --- Define the layers along the Z-axis (beam path) ---
# Define the beam path as a cylinder
beam_radius = 20 # mm
beam_mask = np.sqrt(X**2 + Y**2) <= beam_radius

# Define the start and end of each material layer in mm
bone_z_end = -120 # mm
water_z_end = -70 # mm
soft_z_end = -20 # mm
bone_z_end2 = 20 # mm

# 1. Place a BONE 
bone_mask = (Z >= -150) & (Z < bone_z_end - 5)
materialGrid[beam_mask & bone_mask] = BONE

# 2. Place a WATER layer
water_mask = (Z >= bone_z_end) & (Z < water_z_end - 5)
materialGrid[beam_mask & water_mask] = WATER

# 3. Place a SOFT tissue layer
soft_mask = (Z >= water_z_end) & (Z < soft_z_end - 5)
materialGrid[beam_mask & soft_mask] = SOFT

# 4. Place a BONE layer
bone_mask2 = (Z >= soft_z_end) & (Z < bone_z_end2)
materialGrid[beam_mask & bone_mask2] = BONE

# Save the material grid to a file for python and C++ usage
np.save("./materialGrid.npy", materialGrid)

# Flip the z-axis (axis=2 in your original grid)
materialGrid_flipped = materialGrid[:, :, ::-1]
materialGridTOPAS = np.transpose(materialGrid_flipped, (2, 0, 1))
materialGridTOPAS.flatten(order='C').astype(np.int16).tofile("materialGrid.dat")

# --- Visualization of the layered phantom ---
# Your colors and labels for the materials
colors = ['cyan', 'blue', 'brown', 'pink']
labels = ['Lung', 'Water', 'Bone', 'Soft']

# Create the colormap and norm
cmap = mcolors.ListedColormap(colors)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

fig, axs = plt.subplots(1, 3, figsize=(21, 7))

# Helper function to add a colorbar
def add_colorbar(mappable, ax, labels):
    cbar = fig.colorbar(mappable, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(labels)
    # cbar.set_label('Material ID')

# --- Axial Slice (XY plane) ---
# Select a slice in the middle of the water layer for a clear view
zSlice_index = 50
zSlice = materialGrid[:, :, zSlice_index]
im0 = axs[0].imshow(
    zSlice,
    cmap=cmap,
    norm=norm,
    extent=[x[0], x[-1], y[0], y[-1]],
    origin='lower',
    aspect='equal'  # Ensures a 1:1 aspect ratio
)
axs[0].set_title(f'Axial Slice at Z = {z[zSlice_index]:.0f} mm')
axs[0].set_xlabel('X (mm)')
axs[0].set_ylabel('Y (mm)')
add_colorbar(im0, axs[0], labels)

# --- Sagittal Slice (YZ plane) ---
# Take the central slice where X is 0 mm
xSlice = materialGrid[xBins // 2 - 1, :, :]
im1 = axs[1].imshow(
    xSlice.T, 
    cmap=cmap,
    norm=norm,
    extent=[y[0], y[-1], z[0], z[-1]],
    origin='lower',
    aspect='equal'  # Ensures the correct physical aspect ratio
)
axs[1].set_title(f'Sagittal Slice at X = {x[xBins // 2]:.0f} mm')
axs[1].set_xlabel('Y (mm)')
axs[1].set_ylabel('Z (mm)')
add_colorbar(im1, axs[1], labels)

# --- Coronal Slice (XZ plane) ---
# Take the central slice where Y is 0 mm
ySlice = materialGrid[:, yBins // 2 - 1, :]
im2 = axs[2].imshow(
    ySlice.T,
    cmap=cmap,
    norm=norm,
    extent=[x[0], x[-1], z[0], z[-1]],
    origin='lower',
    aspect='equal'  # Ensures the correct physical aspect ratio
)
axs[2].set_title(f'Coronal Slice at Y = {y[yBins // 2]:.0f} mm')
axs[2].set_xlabel('X (mm)')
axs[2].set_ylabel('Z (mm)')
add_colorbar(im2, axs[2], labels)

plt.tight_layout()
plt.savefig("LayeredMaterialGridSlices.pdf")
plt.close()