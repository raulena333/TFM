import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Matplotlib params
params = {
    'xtick.labelsize': 18,    
    'ytick.labelsize': 18,      
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'legend.fontsize': 12
}
pylab.rcParams.update(params)  # Apply changes


# --- User-defined variables ---
# Set the path to your data file
data_file = './PlotsBetter/ProfileMeanEnergy_XIndex25_YIndex25.txt'

# Set the path where you want to save the output PDF
save_path = './PlotsBetter/'
x_index = 25
y_index = 25

# --- Data Loading ---
try:
    # Load the data from the text file.
    # The file should have four columns separated by spaces or tabs.
    data = np.loadtxt(data_file)
    print(f"Successfully loaded data from {data_file}")
except FileNotFoundError:
    print(f"Error: The file '{data_file}' was not found.")
    print("Please make sure the file exists in the same directory as the script.")
    exit()

# Extract the data columns
# The first column is the independent variable (Z), the other three are the profiles.
zRange = data[:, 0]
profileZMeanTOPAS = data[:, 1]
profileZMeanTrans = data[:, 2]
profileZMeanNorm = data[:, 3]

# --- Plotting ---
# Create the main plot
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

# Calculate the differences for the inset plot
diffMeanTrans = profileZMeanTOPAS - profileZMeanTrans
diffMeanNorm = profileZMeanTOPAS - profileZMeanNorm

ax_inset.plot(zRange, diffMeanTrans, label='TOPAS - Transformation', color='blue', linestyle='-', linewidth=1)
ax_inset.plot(zRange, diffMeanNorm, label='TOPAS - Normalization', color='green', linestyle='-', linewidth=1)

ax_inset.set_xlabel('Z (mm)', fontsize=13)
#ax_inset.set_xlim(-150, -100)
ax_inset.set_ylabel('Difference (MeV)', fontsize=12)
ax_inset.tick_params(labelsize=11)
ax_inset.grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(f'{save_path}ProfileMeanEnergy_XIndex{x_index}_YIndex{y_index}.pdf')
plt.close()

print(f"Plot saved as ProfileMeanEnergy_XIndex{x_index}_YIndex{y_index}.pdf")