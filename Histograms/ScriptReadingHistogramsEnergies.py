import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import os

# Matplotlib params
params = {
    'xtick.labelsize': 16,    
    'ytick.labelsize': 16,      
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'legend.fontsize': 16
}
pylab.rcParams.update(params)  # Apply changes


def returnDoseEnergyValue(energyFile, doseFile):
    
    if not os.path.exists(energyFile):
        print(f"Error: File {energyFile} does not exist.")
        return None, None
    
    if not os.path.exists(doseFile):
        print(f"Error: File {doseFile} does not exist.")
        return None, None

    # Load data, skipping first 5 rows
    dataEnergy = np.loadtxt(energyFile, skiprows=5)
    dataDose = np.loadtxt(doseFile, skiprows=5)
    
    return dataEnergy, dataDose

def calculateEnergyLossDose(energyValues, initialEnergy, mass):

    energyLoss = 0 # MeV
    for energy in energyValues:
        loss = initialEnergy - energy
        energyLoss += loss 
        
    energyJ = 1.6022e-13 * energyLoss # J / Kg
        
    return energyLoss, energyJ / mass


# Energy values
energies = [200, 9]

savePath = "./PlotsEnergies/"

# Create directory if it doesn't exist
if not os.path.exists(savePath):
    os.makedirs(savePath)

# Number of bins for histograms
numberOfBinsAngles = 100

# Load data from files
for i, energy in enumerate(energies):   
    fileName = f"OutputVoxel{energy}MeV.phsp"
    
    try:
        discardedData = 0
        newData = np.loadtxt(fileName)
        print(f'{fileName} loaded successfully.')

        # Extract relevant columns
        finalEnergy, initialEnergy = newData[:, [5,10]].T
        if energy == 200:
            mask = finalEnergy > 195
        if energy == 175:
            mask = finalEnergy > 170
        if energy == 150:
            mask = finalEnergy > 145
        if energy == 125:
            mask = finalEnergy > 120
        if energy == 100:
            mask = finalEnergy > 95
        if energy == 75:
            mask = finalEnergy > 70
        if energy == 50:
            mask = finalEnergy > 45
        if energy == 25:
            mask = finalEnergy > 20
        if energy == 15:
            mask = finalEnergy > 10
        else:
            mask = np.ones_like(finalEnergy, dtype=bool)
        finalEnergy = finalEnergy[mask]
        initialEnergy = initialEnergy[mask]
        logNormLoss = np.log((initialEnergy - finalEnergy) / initialEnergy) 
        logNormLossSqrt =  logNormLoss / np.sqrt(initialEnergy)
        logNormLossLnE = logNormLoss / np.log(initialEnergy)
        logNormLossLnSqrt = logNormLoss / np.log(np.sqrt(initialEnergy))
        
        # Plot histograms
        # Normalized Loss
        fig1 = plt.figure(figsize=(7.25, 6))
        sns.histplot(logNormLoss, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False)
        plt.xlabel(r'$ln(\dfrac{E_i-E_f}{E_i})$ (u.a.)')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{savePath}OutputHistograms{energy}MeV_NormLossEnergy.pdf')
        plt.close(fig1)

        # Sqrt Energy
        fig2 = plt.figure(figsize=(7.25, 6))
        sns.histplot(logNormLossSqrt, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False)
        plt.xlabel(r'$\dfrac{ln(\frac{E_i-E_f}{E_i})}{\sqrt{E_i}}$ (MeV$^{-1/2}$)')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{savePath}OutputHistograms{energy}MeV_NormLossSqrtEnergy.pdf')
        plt.close(fig2)

        # Ln Energy
        fig3 = plt.figure(figsize=(7.25, 6))
        sns.histplot(logNormLossLnE, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False)
        plt.xlabel(r'$\dfrac{ln(\frac{E_i-E_f}{E_i})}{ln(E_i)}$ (MeV$^{-1}$)')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{savePath}OutputHistograms{energy}MeV_NormLossLnEnergy.pdf')
        plt.close(fig3)
        
        fig4 = plt.figure(figsize=(7.25, 6))
        sns.histplot(logNormLossLnSqrt, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False)
        plt.xlabel(r'$\dfrac{ln(\frac{E_i-E_f}{E_i})}{ln(\sqrt{E_i})}$ (MeV$^{-1/2}$)')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{savePath}OutputHistograms{energy}MeV_NormLossLnSqrtEnergy.pdf')
        plt.close(fig4)

    except Exception as e:
        print(f'Error loading {fileName}: {e}')