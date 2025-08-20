import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pylab as pylab
import seaborn as sns
import argparse
import os
import matplotlib.colors as colors

# Matplotlib params
params = {
    'xtick.labelsize': 15,    
    'ytick.labelsize': 15,      
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'legend.fontsize': 15
}
pylab.rcParams.update(params)  # Apply changes

# Energy values
energies = [200, 12]
savePath = "./Plots/NoThreshold/"

# Create directory if it doesn't exist
if not os.path.exists(savePath):
    os.makedirs(savePath)

# Number of bins for histograms
numberOfBinsAngles = 100
numberOfBinsEnergies = 100
jointNumberOfBins = 100

energyDosePath = f'./EnergyDoseFiles/'


# Load data from files
for i, energy in enumerate(energies):   
    fileName = f"OutputVoxel{energy}MeV.phsp"
    
    try:
        newData = np.loadtxt(fileName)
        print(f'{fileName} loaded successfully.')

        # Extract relevant columns
        finalDirectionCosineX, finalDirectionCosineY, finalEnergy, isSign, initialEnergy = newData[:, [3,4,5,8,10]].T
        initialData = len(initialEnergy)
        
        if energy == 200:
            mask = finalEnergy > 195
        elif energy == 175:
            mask = finalEnergy > 170
        elif energy == 150:
            mask = finalEnergy > 145
        elif energy == 125:
            mask = finalEnergy > 120
        elif energy == 100:
            mask = finalEnergy > 95
        elif energy == 75:
            mask = finalEnergy > 70
        elif energy == 50:
            mask = finalEnergy > 45
        elif energy == 25:
            mask = finalEnergy > 20
        else:
            mask = np.ones_like(finalEnergy, dtype=bool)
        finalDirectionCosineX = finalDirectionCosineX[mask]
        finalDirectionCosineY = finalDirectionCosineY[mask]
        finalEnergy = finalEnergy[mask]
        initialEnergy = initialEnergy[mask]
        isSign = isSign[mask]

        # Compute final angles
        finalAngles = []
        for j, (directionX, directionY, sign) in enumerate(zip(finalDirectionCosineX, finalDirectionCosineY, isSign)):
            value = 1 - directionX**2 - directionY**2
            value = np.maximum(value, 0)
            directionZ = np.sqrt(value)

            if sign == 0:
                directionZ *= -1   
            angle = np.degrees(np.arccos(directionZ))
            finalAngles.append(angle)
        
        # Plot histograms
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))

        sns.histplot(finalEnergy, bins=numberOfBinsEnergies, edgecolor="black", color='orange', kde=False, ax=axs[0])
        axs[0].set_xlabel(r'Final Energy (MeV)')
        #axs[0].set_title('Final Energy distribution')
        axs[0].set_yscale('log')
         
        sns.histplot(finalAngles, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False, ax=axs[1])
        axs[1].set_xlabel(r'$\theta$ (deg)')
        #axs[1].set_title('Final Angles distribution')
        axs[1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(f'{savePath}OutputHistograms{energy}MeV_Raw.pdf')
        plt.close(fig)

        # Compute 2D Histogram
        hist1, xedges1, yedges1 = np.histogram2d(finalAngles, finalEnergy, bins=jointNumberOfBins)
        finalProbabilities = hist1 / np.sum(hist1)

        fig2, axs2 = plt.subplots(figsize=(12, 8))
        h1 = axs2.pcolormesh(xedges1, yedges1, finalProbabilities.T, cmap='Reds', shading='auto')
        fig2.colorbar(h1, ax=axs2, label='Probability')
        axs2.set_xlabel(r'$\theta$ (deg)')
        axs2.set_ylabel(r'Final Energy (MeV)')
        
        plt.savefig(f'{savePath}Output2DHistograms{energy}MeV_Raw.pdf')
        plt.close(fig2) 

    except Exception as e:
        print(f'Error loading {fileName}: {e}')