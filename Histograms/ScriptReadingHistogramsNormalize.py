import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pylab as pylab
import seaborn as sns
import os
from scipy.interpolate import interp1d

# Matplotlib params
params = {
    'xtick.labelsize': 17,    
    'ytick.labelsize': 17,      
    'axes.titlesize': 17,
    'axes.labelsize': 17,
    'legend.fontsize': 17
}
pylab.rcParams.update(params)  # Apply changes

# Energy values
energies = [200, 9]

# Path for saving plots
savePath = "./PlotsNormalize/"

# Create directory if it doesn't exist
if not os.path.exists(savePath):
    os.makedirs(savePath)
    
# Number of bins for histograms
numberOfBinsAngles = 100
numberOfBinsEnergies = 100
jointNumberOfBins = 100


# Define known energy-threshold pairs (from your manual logic)
knownInitialEnergies = np.array([200, 175, 150, 125, 100, 75, 50, 25, 15])
knownThresholds = np.array([195, 170, 145, 120, 95, 70, 45, 20, 10])

# Create an interpolation function
thresholdFn = interp1d(knownInitialEnergies, knownThresholds,
                                kind='linear', fill_value='extrapolate')

# Load data from files
for i, energy in enumerate(energies):   
    fileName = f"OutputVoxel{energy}MeV.phsp"

    try:
        discardedData = 0
        newData = np.loadtxt(fileName)
        print(f'{fileName} loaded successfully.')

        # Extract relevant columns
        finalDirectionCosineX, finalDirectionCosineY, finalEnergy, isSign, initialEnergy = newData[:, [3,4,5,8,10]].T

        if energy >15:
            # Then inside your loop:
            thresholdEnergy = thresholdFn(energy)
            mask = finalEnergy > thresholdEnergy
        else:
            mask = np.ones_like(finalEnergy, dtype=bool)

        finalDirectionCosineX = finalDirectionCosineX[mask]
        finalDirectionCosineY = finalDirectionCosineY[mask]
        finalEnergy = finalEnergy[mask]
        initialEnergy = initialEnergy[mask]
        isSign = isSign[mask]
        
        # Normalize energy values
        finalEnergyMin = np.min(finalEnergy)
        finalEnergyMax = np.max(finalEnergy)
        efNorm = (finalEnergy - finalEnergyMin) / (finalEnergyMax - finalEnergyMin) # between 0 and 1

        # Compute final angles
        finalAngles = []
        maxTheta = 0

        for j, (directionX, directionY, sign) in enumerate(zip(finalDirectionCosineX, finalDirectionCosineY, isSign)):
            value = 1 - directionX**2 - directionY**2
            value = np.maximum(value, 0)
            directionZ = np.sqrt(value)

            if sign == 0:
                directionZ *= -1   
            angle = np.degrees(np.arccos(directionZ))
            finalAngles.append(angle)
      
        maxTheta = np.max(finalAngles)
        angleNorm = np.array(finalAngles) / maxTheta # between 0 and 1
        
        # Plot histograms
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))

        sns.histplot(efNorm, bins=numberOfBinsEnergies, edgecolor="black", color='orange', kde=False, ax=axs[0])
        axs[0].set_xlabel(r'Normalized Energy (a.u.)')
        #axs[0].set_title('Final Energy distribution')
        axs[0].set_yscale('log')
         
        sns.histplot(angleNorm, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False, ax=axs[1])
        axs[1].set_xlabel('Normalized Angle (a.u.)')
        #axs[1].set_title('Final Angles distribution')
        axs[1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(f'{savePath}OutputHistograms{energy}MeV_Normalized.pdf')
        plt.close(fig)
        
        fig1, axs1 = plt.subplots(figsize=(7.25, 6))
        sns.histplot(angleNorm, bins=numberOfBinsEnergies, edgecolor="black", color='orange', kde=False, ax=axs1)
        axs1.set_xlabel(r'Normalized Angle (a.u.)')
        # axs1.set_title('Final Angles distribution')
        axs1.set_yscale('log')

        plt.tight_layout()        
        plt.savefig(f'{savePath}OutputHistogramAngle{energy}MeV_NormalizedAngle.pdf')
        plt.close(fig1)

        # Compute 2D Histogram
        hist1, xedges1, yedges1 = np.histogram2d(angleNorm, efNorm, bins=jointNumberOfBins)
        finalProbabilities = hist1 / np.sum(hist1)
        
        # Avoid log(0) by adding a small constant and then converting to dB
        log_probabilities_dB = 10 * np.log10(finalProbabilities + 1e-12)

        fig2, axs2 = plt.subplots(figsize=(8, 6))
        h1 = axs2.pcolormesh(xedges1, yedges1, log_probabilities_dB.T, cmap='Reds', shading='auto')
        fig2.colorbar(h1, ax=axs2, label='Probability')
        axs2.set_xlabel('Normalized Angle (a.u.)')
        axs2.set_ylabel(r'Normalized Energy (a.u.)')

        plt.tight_layout()
        plt.savefig(f'{savePath}Output2DHistograms{energy}MeV_Normalized.pdf')
        plt.close(fig2) 
        
        fig3, axs3 = plt.subplots(figsize=(7.25, 6))
        sns.histplot(efNorm, bins=numberOfBinsEnergies, edgecolor="black", color='orange', kde=False, ax=axs3)
        axs3.set_xlabel(r'Normalized Energy (a.u.)')
        # axs3.set_title('Final Energy distribution')
        axs3.set_yscale('log')

        plt.tight_layout()        
        plt.savefig(f'{savePath}OutputHistogramEnergy{energy}MeV_NormalizedEnergy.pdf')
        plt.close(fig3)

    except Exception as e:
        print(f'Error loading {fileName}: {e}')