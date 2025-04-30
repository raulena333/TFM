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
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,      
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'legend.fontsize': 14
}
pylab.rcParams.update(params)  # Apply changes

if __name__ == "__main__":

    savePath = "./Plots/"

    # Create directory if it doesn't exist
    if not os.path.exists(savePath):
        os.makedirs(savePath)
        
    # Number of bins for histograms
    numberOfBinsAngles = 100
    numberOfBinsEnergies = 100
    jointNumberOfBins = 100

    # Load data from files  
    fileName = f"BigBoxSimulationTOPAS.phsp"
        
    discardedData = 0
    newData = np.loadtxt(fileName)
    print(f'{fileName} loaded successfully.')

    # Extract relevant columns
    finalDirectionCosineX, finalDirectionCosineY, finalEnergy, isSign, initialEnergy = newData[:, [3,4,5,8,10]].T
    logFinalEnergy = np.log((initialEnergy - finalEnergy) / initialEnergy) / np.sqrt(initialEnergy)
    energy = np.mean(initialEnergy)
    
    threshold = [-0.033, -0.05, 300] # maxEnergy, minEnergy, angleThreshold
    uniformMaxEnergy = threshold[0]
    uniformMinEnergy = threshold[1]
    uniformAngleThreshold = threshold[2]
    
    # No filtering
    mask = (logFinalEnergy < uniformMaxEnergy) & (logFinalEnergy > uniformMinEnergy)       
    filteredFinalDirectionCosineX = finalDirectionCosineX[mask]
    filteredFinalDirectionCosineY = finalDirectionCosineY[mask]
    filteredIsSign = isSign[mask]
    logFinalEnergy = logFinalEnergy[mask]
    
    # Compute final angles
    finalAngles = []
    indexToDelete = []

    for j, (directionX, directionY, sign) in enumerate(zip(filteredFinalDirectionCosineX, filteredFinalDirectionCosineY, filteredIsSign)):
        value = 1 - directionX**2 - directionY**2
        value = np.maximum(value, 0)
        directionZ = np.sqrt(value)

        if sign == 0:
            directionZ *= -1   
        angle = np.degrees(np.arccos(directionZ))
        angle *= np.sqrt(energy)
        
        if angle > uniformAngleThreshold:
            indexToDelete.append(j)
            discardedData += 1
        else:
            finalAngles.append(angle)
            
    if indexToDelete:
        indexToDelete = np.array(indexToDelete, dtype=int)
        logFinalEnergy = np.delete(logFinalEnergy, indexToDelete)
        
    percentajeDiscarded = discardedData / len(finalEnergy) * 100
    print("Percentage of discarded energy: ", percentajeDiscarded)
               
    # Plot histograms
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    sns.histplot(logFinalEnergy, bins=numberOfBinsEnergies, edgecolor="black", color='orange', kde=False, ax=axs[0]) 
    axs[0].set_xlabel(r'$\frac{ln((E_i-E_f)/E_i)}{\sqrt{E_i}}$ (MeV$^{-1/2}$)')
    axs[0].set_title('Final Energy distribution')
    axs[0].set_yscale('log')
            
    sns.histplot(finalAngles, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False, ax=axs[1])
    axs[1].set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
    axs[1].set_title('Final Angles distribution')
    axs[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{savePath}BigVoxelHistograms.pdf')
    plt.close(fig)

    
    hist1, xedges1, yedges1 = np.histogram2d(finalAngles, logFinalEnergy, bins=jointNumberOfBins)
    finalProbabilities = hist1 / np.sum(hist1)

    fig2, axs2 = plt.subplots(figsize=(8, 6))
    h1 = axs2.pcolormesh(xedges1, yedges1, finalProbabilities.T, cmap='Reds', shading='auto')
    fig2.colorbar(h1, ax=axs2, label='Probability')
    axs2.set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
    axs2.set_ylabel(r'$ln((E_i-E_f)/E_i)\sqrt{E_i}$ (MeV$^{-1/2}$)')

    plt.tight_layout()
    plt.savefig(f'{savePath}2DBigVoxelHistograms.pdf')
    plt.close(fig2) 
    