import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pylab as pylab

params = {
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,      
    'axes.titlesize' : 14,
    'axes.labelsize' : 14,
    'legend.fontsize': 14
}
pylab.rcParams.update(params)  # Apply changes

# Load data from phsp file and filtr it so we only tanke the proton particle "PDG Format = 2212"
data = np.loadtxt('OutputPosition.phsp')
fileteredData = data[data[:, 7] == 2212]
finalDirectionCosineX, finalDirectionCosineY, finalEnergy, isSign, initialEnergy, initialDirectionCosineZ = fileteredData[:, [3,4,5,8,10,16]].T

# Calculate initial and final Angles for tables, using isSign to determine whether its + or - 
# For calculating the final direction along de Z-axis, we will use the Xcosine and Ycosine directions using the equaiton : ±\sqrt(1-d_x^2-d_y^2)
# This formula comes from the folliwing web page: https://mathworld.wolfram.com/DirectionCosine.html
finalAngles = []
for directionX, directionY, sign in zip(finalDirectionCosineY, finalDirectionCosineX, isSign):
    directionZ = np.sqrt(1 - directionX * directionX - directionY * directionY)
    if sign == 1:
        angle = np.arccos(directionZ)
    else:
        angle = np.arccos(- directionZ)     
    finalAngles.append(np.degrees(angle))

# Calculate initial direction aloong the Z-axis
initialAngles = []
for directionZ in initialDirectionCosineZ:
    angle = np.arccos(directionZ)
    initialAngles.append(np.degrees(angle))

# Combine original data with the angles columns
newData = np.column_stack((initialEnergy, initialAngles, finalEnergy, finalAngles))

# Plot results for visulization as histograms
fig, axs = plt.subplots(2, 2, figsize=(10, 8.33))

axs[0, 0].hist(initialEnergy, bins = 30, color = 'blue', edgecolor = "black", alpha = 0.6, label = 'Initial Energy (discrete)')
axs[0, 0].set_xlabel('Energy (MeV)')
axs[0, 0].set_title('Initial Energy distribution')

axs[0, 1].hist(initialAngles, bins = 30, color = 'green', alpha = 0.6, label = 'Initial Angles')
axs[0, 1].set_xlabel('Angle (deg)')
axs[0, 1].set_title('Initial Angles distribution')

axs[1, 0].hist(finalEnergy, bins = 30, color = 'orange', alpha = 0.6, label = 'Final Energy')
axs[1, 0].set_xlabel('Energy (MeV)')
axs[1, 0].set_title('Final Energy distribution')

axs[1, 1].hist(finalAngles, bins = 30, color = 'red', alpha = 0.6, label = 'Final Angles')
axs[1, 1].set_xlabel('Angle (deg)')
axs[1, 1].set_title('Final Angles distribution')

plt.tight_layout()
plt.show()