import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# Matplotlib params
params = {
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,      
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'legend.fontsize': 14
}
pylab.rcParams.update(params)  # Apply changes  

materialsPath = "./Materials/"
materials = ["Water", "AirDry", "Tissue", "Bone"]

# Data preparation
x_data = {}
y_data = {"Stopping Power": {}, "Range": {}}

# Load the data for all materials
for material in materials:
    data = np.loadtxt(materialsPath + material + ".txt", skiprows=1)
    x_data[material] = data[:, 0]
    y_data["Stopping Power"][material] = data[:, 1]
    y_data["Range"][material] = data[:, 2]

# Plotting
fig = plt.figure(figsize=(11, 6))

# Plot Stopping Power on the first subplot (ax1)
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height]
for material in materials:
    ax1.plot(x_data[material], y_data["Stopping Power"][material], label=material, linestyle='-')
ax1.set_xlabel("Proton Energy [MeV]")
ax1.set_ylabel(r"$\frac{-dE}{dx}$ [MeV$\cdot$cm$^2$/g]")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.grid(True)
ax1.legend(loc=(0.8, 0.7))

# Plot Range on the second subplot (ax2)
ax2 = fig.add_axes([0.2, 0.16, 0.33, 0.34])  # [left, bottom, width, height]
for material in materials:
    ax2.plot(x_data[material], y_data["Range"][material], label=material,  linestyle='-')
#ax2.set_xlabel("Proton Energy [MeV]")
ax2.set_ylabel(r"Range [g$\cdot$cm$^2$]")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.grid(True)
# ax2.legend()

# Save the figure with both subplots
plt.savefig("StoppingPowerAndRangePlot.pdf")
plt.show()
