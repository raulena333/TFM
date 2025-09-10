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
pylab.rcParams.update(params) # Apply changes

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
    fileName = f"OutputVoxelSheet.phsp"

    # Thresholds for transformed energy and angular filtering
    threshold = [0, -0.6, 70] # [maxLogE, minLogE, maxAngleDeg]

    # Load data
    data = np.loadtxt(fileName)
    print(f'{fileName} loaded successfully.')

    finalDirX, finalDirY, finalEnergy, isSign, initialEnergy = data[:, [3, 4, 5, 8, 10]].T
    energy = np.mean(initialEnergy)

    # ---------- TRANSFORMED ENERGY PATH ----------
    logE = np.log((initialEnergy - finalEnergy) / initialEnergy)
    logE *= 1 / np.sqrt(initialEnergy)

    # Clamp logE to the valid range
    logE = np.clip(logE, threshold[1], threshold[0])

    maskTrans = (logE < threshold[0]) & (logE > threshold[1])
    logE_T = logE[maskTrans]
    dirX_T = finalDirX[maskTrans]
    dirY_T = finalDirY[maskTrans]
    sign_T = isSign[maskTrans]
    finalEnergy = finalEnergy[maskTrans]

    dirZ_T = np.sqrt(np.clip(1 - dirX_T**2 - dirY_T**2, 0, 1))
    dirZ_T[sign_T == 0] *= -1
    angle = np.degrees(np.arccos(np.clip(dirZ_T, -1.0, 1.0)))
    angle_T = angle * np.sqrt(energy)

    # Clam angle to the valid range
    angle_T = np.clip(angle_T, 0, threshold[2])

    # Apply angular cutoff
    angle_mask = angle_T <= threshold[2]
    angle = angle[angle_mask]
    angle_T = angle_T[angle_mask]
    logE_T = logE_T[angle_mask]
    finalEnergy = finalEnergy[angle_mask]
    
    # --- SAVE THE PROCESSED DATA FOR REFERENCE ---
    save_reference_data = True  # Set to True to save the data
    reference_file_name = f'{savePath}topasReferenceData.npz'

    if save_reference_data:
        np.savez(reference_file_name, energies=finalEnergy, angles=angle)
        print(f"TOPAS reference data saved to {reference_file_name}")

    # ---------- NORMALIZED ENERGY PATH ----------
    maxTheta = np.max(angle)
    minTheta = np.min(angle)
    angle_N_norm = (angle - minTheta) / (maxTheta - minTheta)

    # Normalize final energy for histogram
    finalEnergyMin = np.min(finalEnergy)
    finalEnergyMax = np.max(finalEnergy)
    energy_Norm = (finalEnergy - finalEnergyMin) / (finalEnergyMax - finalEnergyMin)

    # Plot histograms
    # --- ANGLE HISTOGRAM (normalizado al máximo) ---
    angle = angle[angle < 3]
    
    angleHist, angleBins = np.histogram(angle, bins=numberOfBinsAngles)
    angleCenters = 0.5 * (angleBins[:-1] + angleBins[1:])
    angleHistNorm = angleHist / np.max(angleHist)

    # Buscar el ángulo donde la frecuencia cae a ≤ 1%
    threshold = 1 / 100
    angleBelowThreshold = np.where(angleHistNorm <= threshold)[0]
    angleCutoff = angleCenters[angleBelowThreshold[0]] if len(angleBelowThreshold) > 0 else None

    plt.figure(figsize=(7.25, 6))
    # Use seaborn.histplot for plotting the histogram from raw data
    sns.histplot(data=angle, bins=numberOfBinsAngles, color='darkred', stat='probability')
    plt.xlabel("Angle (º)")
    plt.ylabel("Probability")
    plt.yscale("linear")
    plt.title(f"Angle Distribution - Normalized - E = {energy:.2f} MeV")
    if angleCutoff is not None:
        plt.axvline(angleCutoff, color='blue', linestyle='--', label=f"1% max at {angleCutoff:.2f}°", linewidth=0.5)
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{savePath}SamplingAngleNorm_Voxel_Energy{energy:.2f}.pdf")
    plt.close()

    # --- ENERGY HISTOGRAM (normalizado al máximo) ---
    # Apply a mask to filter the data
    finalEnergy = finalEnergy[finalEnergy > 144.4]

    # Re-calculate histogram and normalized frequency for the filtered data
    energyHist, energyBins = np.histogram(finalEnergy, bins=numberOfBinsEnergies)
    energyCenters = 0.5 * (energyBins[:-1] + energyBins[1:])
    energyHistNorm = energyHist / np.max(energyHist)

    energyBelowThreshold = np.where(energyHistNorm <= threshold)[0]
    energyCutoff = energyCenters[energyBelowThreshold[0]] if len(energyBelowThreshold) > 0 else None

    plt.figure(figsize=(7.25, 6))
    # Use seaborn.histplot for plotting the histogram from raw data
    sns.histplot(data=finalEnergy, bins=numberOfBinsEnergies, color='darkred', stat='probability')
    plt.xlabel("Final energy (MeV)")
    plt.ylabel("Probability")
    plt.yscale("linear")
    plt.title(f"Final Energy Distribution - Normalized - E = {energy:.2f} MeV")
    if energyCutoff is not None:
        plt.axvline(energyCutoff, color='blue', linestyle='--', label=f"1% max at {energyCutoff:.2f} MeV", linewidth=0.5)
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{savePath}SamplingEnergyNorm_Voxel_Energy{energy:.2f}.pdf")
    plt.close()

    # --- STATISTICS ---
    print(f'Mean energy: {np.mean(finalEnergy):.4f}')
    print(f'Std energy:  {np.std(finalEnergy):.4f}')
    print(f'Mean angle:  {np.mean(angle):.4f}')
    print(f'Std angle:   {np.std(angle):.4f}')
    if angleCutoff is not None:
        print(f"Angle at which freq drops to 1%: {angleCutoff:.2f}°")
    if energyCutoff is not None:
        print(f"Energy at which freq drops to 1%: {energyCutoff:.2f} MeV")
