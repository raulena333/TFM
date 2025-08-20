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

savePath = "./PlotsAngles/"

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
        finalDirectionCosineX, finalDirectionCosineY, finalEnergy, isSign, initialEnergy = newData[:, [3,4,5,8,10]].T
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
        finalDirectionCosineX = finalDirectionCosineX[mask]
        finalDirectionCosineY = finalDirectionCosineY[mask]
        finalEnergy = finalEnergy[mask]
        initialEnergy = initialEnergy[mask]
        isSign = isSign[mask]
        
        # Compute final angles
        finalAnglesE = []
        finalAnglesSqrtE = []
        finalAnglesEE = []
        finalAnglesLnE = []

        for j, (directionX, directionY, sign) in enumerate(zip(finalDirectionCosineX, finalDirectionCosineY, isSign)):
            value = 1 - directionX**2 - directionY**2
            value = np.maximum(value, 0)
            directionZ = np.sqrt(value)

            if sign == 0:
                directionZ *= -1   
            angle = np.degrees(np.arccos(directionZ))
            finalAnglesE.append(angle * initialEnergy[j])
            finalAnglesSqrtE.append(angle * np.sqrt(initialEnergy[j]))
            finalAnglesEE.append(angle * initialEnergy[j]**2)
            finalAnglesLnE.append(angle * np.log(initialEnergy[j]))
        
        # Plot histograms
        # Dot Energy
        fig1 = plt.figure(figsize=(7.25, 6))
        sns.histplot(finalAnglesE, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False)
        plt.xlabel(r'Angle$\cdot E_i$ (deg$\cdot$MeV)')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{savePath}OutputHistograms{energy}MeV_DotEnergyAngles.pdf')
        plt.close(fig1)

        # Sqrt Energy
        fig2 = plt.figure(figsize=(7.25, 6))
        sns.histplot(finalAnglesSqrtE, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False)
        plt.xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{savePath}OutputHistograms{energy}MeV_SqrtEnergyAngles.pdf')
        plt.close(fig2)

        # Energy Squared
        fig3 = plt.figure(figsize=(7.25, 6))
        sns.histplot(finalAnglesEE, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False)
        plt.xlabel(r'Angle$E_i^2$ (deg$\cdot$MeV$^2$)')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{savePath}OutputHistograms{energy}MeV_EnergySquaredAngles.pdf')
        plt.close(fig3)

        # Ln Energy
        fig4 = plt.figure(figsize=(7.25, 6))
        sns.histplot(finalAnglesLnE, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False)
        plt.xlabel(r'Angle$ln(E_i)$ (deg$\cdot$MeV)')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{savePath}OutputHistograms{energy}MeV_LnEnergyAngles.pdf')
        plt.close(fig4)


    except Exception as e:
        print(f'Error loading {fileName}: {e}')