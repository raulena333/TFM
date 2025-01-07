import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import matplotlib.pylab as pylab
import seaborn as sns

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
    
# 

# Combine original data with the angles columns
newData = np.column_stack((initialEnergy, initialAngles, finalEnergy, finalAngles))

# Plot results for visulization as histograms
fig, axs = plt.subplots(2, 2, figsize=(10, 8.33))

sns.histplot(initialEnergy, bins=30, color='blue', kde=False, ax=axs[0, 0], edgecolor="black",)
axs[0, 0].set_xlabel('Energy (MeV)')
axs[0, 0].set_title('Initial Energy distribution')
#axs[0, 0].set_yscale('log')

sns.histplot(initialAngles, bins=30, edgecolor="black", color='green', kde=False, ax=axs[0, 1])
axs[0, 1].set_xlabel('Angle (deg)')
axs[0, 1].set_title('Initial Angles distribution')
#axs[0, 1].set_yscale('log')

sns.histplot(finalEnergy, bins=30, edgecolor="black", color='orange', kde=False, ax=axs[1, 0])
axs[1, 0].set_xlabel('Energy (MeV)')
axs[1, 0].set_title('Final Energy distribution')
#axs[1, 0].set_yscale('log')

sns.histplot(finalAngles, bins=30, edgecolor="black", color='red', kde=False, ax=axs[1, 1])
axs[1, 1].set_xlabel('Angle (deg)')
axs[1, 1].set_title('Final Angles distribution')
#axs[1, 1].set_yscale('log')

plt.tight_layout()
plt.show()

# Plot results for visulization as 2D histogram as heatmaps
fig1, axs1 = plt.subplots(1, 2, figsize=(16, 6))

h1 = axs1[0].hist2d(initialAngles, initialEnergy, bins=50, cmap='Blues')
fig1.colorbar(h1[3], ax=axs1[0], label='Counts')
axs1[0].set_xlabel('Angle (deg)')
axs1[0].set_ylabel('Energy (MeV)')
axs1[0].set_title('2D Histogram of Initial Energy vs Initial Angle')
#axs1[0].set_xlim(179, 180)  

h2 = axs1[1].hist2d(finalAngles, finalEnergy, bins=50, cmap='Reds')
fig1.colorbar(h2[3], ax=axs1[1], label='Counts')
axs1[1].set_xlabel('Angle (deg)')
axs1[1].set_ylabel('Energy (MeV)')
axs1[1].set_title('2D Histogram of Final Energy vs Final Angle')
#axs1[1].set_xlim(0, 10)  

plt.tight_layout()
plt.show()

# Violin plot to compare initial and final angles
# plt.figure(figsize=(8, 6))
# sns.violinplot(data=[initialAngles, finalAngles], inner="point", palette="muted")
# plt.xticks([0, 1], ['Initial Angles', 'Final Angles'])
# plt.ylabel('Angle (degrees)')
# plt.title('Comparison of Initial and Final Angles')
# plt.show()

# Interactive 3D plot of Initial Angle, Final Angle, and Final Energy
fig = px.scatter_3d(x=initialAngles, y=finalAngles, z=finalEnergy, 
                    labels={'x': 'Initial Angle (degrees)', 'y': 'Final Angles', 'z': 'Final Energy'},
                    title='3D Scatter Plot of Energy vs Angles')
fig.show()
