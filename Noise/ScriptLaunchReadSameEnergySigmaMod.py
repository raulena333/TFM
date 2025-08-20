import re
import os
import time
import struct
import h5py
import numpy as np
from scipy.stats import skew, kurtosis
import subprocess
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

params = {
    'xtick.labelsize': 16,    
    'ytick.labelsize': 16,      
    'axes.titlesize' : 16,
    'axes.labelsize' : 16,
    'legend.fontsize': 16,
}
pylab.rcParams.update(params)  # Apply changes

def modifyBeamEnergy(filePath, newEnergy):
    """
    Modify the beam energy in a TOPAS input file and save it with the same name.

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - newEnergy (float): New energy value to replace in the file.
    """
    patternEnergy = r"(dv:So/MySource/BeamEnergySpectrumValues = 1 )(\d+(\.\d+)?)( MeV)"

    with open(filePath, 'r') as file:
        content = file.read()
    
    # Replace the energy value in the matched line
    updatedFile = re.sub(patternEnergy, rf"dv:So/MySource/BeamEnergySpectrumValues = 1 {newEnergy} MeV", content)
    
    with open(filePath, 'w') as file:
        file.write(updatedFile)

    print(f"Updated beam energy to {newEnergy} MeV in {filePath}.")
    
def modifyInputParameters(filePath, numberOfRuns):
    """
    Modify input parameters, such as the number of runs

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - numberOfRuns (int): Define the number of protons simulated (Number of histories in run).
    """
    # Regular expression pattern to match the line with the number of runs
    pattern = r"(i:So/MySource/NumberOfHistoriesInRun = )(\d+)"
    
    with open(filePath, 'r') as file:
        content = file.read()

    # Replace the number of runs in the matched line
    updatedFile = re.sub(pattern, rf"i:So/MySource/NumberOfHistoriesInRun = {numberOfRuns}", content)
    
    with open(filePath, 'w') as file:
        file.write(updatedFile)

    print(f"Updated number of runs to {numberOfRuns} in {filePath}.")
    
def modifySeedRandomness(filePath, seed):
    """
    Modify input parameters, such as the number of runs

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - seed (int): Define the randomness of the run.
    """
    # Regular expression pattern to match the line with the number of runs
    pattern = r"(i:Ts/Seed = )(\d+)"
    
    with open(filePath, 'r') as file:
        content = file.read()

    # Replace the number of runs in the matched line
    updatedFile = re.sub(pattern, rf"i:Ts/Seed = {seed}", content)
    
    with open(filePath, 'w') as file:
        file.write(updatedFile)

    print(f"Updated seed number in {filePath}.")
    
    
def runTopas(filePath, dataPath):
    """
    Run TOPAS txt-script with the modified input file

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - dataPath (str): Path to the TOPAS G4 data.
    """
    try:
        result = subprocess.run(f'export TOPAS_G4_DATA_DIR={dataPath} && ~/topas/bin/topas {filePath}', 
                        text=True, shell=True)
        if result == 0:
            print("Data loaded and simulation have started succesfully ")
            
    except FileNotFoundError:
        print("TOPAS executable not found. Make sure TOPAS is installed and in your PATH.")
        
def readInputFileSigma(fileName):
    newData = np.loadtxt(fileName)  
    print(f'{fileName} loaded successfully.')
    finalDirectionCosineX, finalDirectionCosineY, finalEnergy, isSign = newData[:, [3,4,5,8]].T
        
    finalAngles = []
    # Loop over each particle's direction cosines
    for directionX, directionY, sign in zip(finalDirectionCosineX, finalDirectionCosineY, isSign):
        directionZ = np.sqrt(1 - directionX**2 - directionY**2)
            
        # Adjust sign of directionZ based on isSign
        if sign == 0:
            directionZ *= -1
        angle = np.degrees(np.arccos(directionZ))
        finalAngles.append(angle)
        
    return finalEnergy, finalAngles


def binNoise(energies, fixedEnergyEdges, angles, fixedAngleEdges):
    # Compute histogram counts for energies
    energyCounts, _ = np.histogram(energies, bins=fixedEnergyEdges)

    # Compute histogram counts for angles
    angleCounts, _ = np.histogram(angles, bins=fixedAngleEdges)
     
    return energyCounts, fixedEnergyEdges, angleCounts, fixedAngleEdges

def calculateMeanAndStd(energyCounts, angleCounts):
    # Convert lists to NumPy arrays for easier calculations
    energyCount = np.array(energyCounts)
    angleCount = np.array(angleCounts)

    # Calculate mean and std across each position (axis=0 means across columns)
    meanEnergy = np.mean(energyCount, axis=0)
    stdEnergy = np.std(energyCount, axis=0)

    meanAngle = np.mean(angleCount, axis=0)
    stdAngle = np.std(angleCount, axis=0)

    return meanEnergy, stdEnergy, meanAngle, stdAngle

def plot2D(meanEnergyBin, stdEnergyBin, meanAngleBin, stdAngleBin, energyEdges, angleEdges, nProton):
    # Calculate coefficient of variation (avoid division by zero)
    cvEnergyBin = np.divide(stdEnergyBin, meanEnergyBin, out=np.zeros_like(stdEnergyBin), where=meanEnergyBin!=0)
    cvAngleBin = np.divide(stdAngleBin, meanAngleBin, out=np.zeros_like(stdAngleBin), where=meanAngleBin!=0)
    
    # Plot Energy mean and std
    fig1, axs1 = plt.subplots(1, 2, figsize=(10, 6))

    axs1[0].bar(energyEdges[:-1], meanEnergyBin, width=np.diff(energyEdges), edgecolor="black", color='orange', align='edge')
    axs1[0].set_xlabel(r'Final Energy (MeV)')
    axs1[0].set_ylabel('Mean Counts')
    axs1[0].set_yscale('log')
    axs1[0].set_title('Mean Energy Distribution per Bin')

    # axs1[1].bar(energyEdges[:-1], stdEnergyBin, width=np.diff(energyEdges), edgecolor="black", color='red', align='edge')
    # axs1[1].set_xlabel(r'Final Energy (MeV)')
    # axs1[1].set_ylabel('Std Dev Counts')
    # #axs1[1].set_yscale('log')
    # axs1[1].set_title('Std Dev Energy per Bin')

    axs1[1].bar(energyEdges[:-1], cvEnergyBin, width=np.diff(energyEdges), edgecolor="black", color='blue', align='edge')
    axs1[1].set_xlabel(r'Final Energy (MeV)')
    axs1[1].set_ylabel('Coefficient of Variation')
    axs1[1].set_title('CV Energy per Bin')

    plt.tight_layout()
    plt.savefig(f'HistogramEnergyStats_{nProton}.pdf')
    plt.close(fig1)

    # Plot Angle mean and std
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 6))

    axs2[0].bar(angleEdges[:-1], meanAngleBin, width=np.diff(angleEdges), edgecolor="black", color='orange', align='edge')
    axs2[0].set_xlabel(r'$\theta$ (deg)')
    axs2[0].set_ylabel('Mean Counts')
    axs2[0].set_yscale('log')
    axs2[0].set_title('Mean Angle Distribution per Bin')

    # axs2[1].bar(angleEdges[:-1], stdAngleBin, width=np.diff(angleEdges), edgecolor="black", color='red', align='edge')
    # axs2[1].set_xlabel(r'$\theta$ (deg)')
    # axs2[1].set_ylabel('Std Dev Counts')
    # #[1].set_yscale('log')
    # axs2[1].set_title('Std Dev Angle per Bin')

    axs2[1].bar(angleEdges[:-1], cvAngleBin, width=np.diff(angleEdges), edgecolor="black", color='blue', align='edge')
    axs2[1].set_xlabel(r'$\theta$ (deg)')
    axs2[1].set_ylabel('Coefficient of Variation')
    axs2[1].set_title('CV Angle per Bin')

    plt.tight_layout()
    plt.savefig(f'HistogramAngleStats_{nProton}.pdf')
    plt.close(fig2)

# def plot2D(meanEnergyBin, stdEnergyBin, meanAngleBin, stdAngleBin, energyEdges, angleEdges, nProton):
#     # Create 2D histogram plot for Energy
#     fig1, axs1 = plt.subplots(1, 2, figsize=(10, 6))

#     # Plot the histogram for the mean energy values
#     axs1[0].bar(energyEdges[:-1], meanEnergyBin, width=np.diff(energyEdges), edgecolor="black", color='orange', align='edge')
#     axs1[0].set_xlabel(r'$E[E_f]$ (MeV)')
#     axs1[0].set_ylabel(r'Counts')
#     # axs1[0].set_title('Final Mean Energy Distribution per Bin')
#     axs1[0].set_yscale('log')

#     # Plot the histogram for the standard deviation of energy
#     axs1[1].bar(energyEdges[:-1], stdEnergyBin, width=np.diff(energyEdges), edgecolor="black", color='red', align='edge')
#     axs1[1].set_xlabel(r'$\sigma_{E_f}$ (MeV)')
#     axs1[1].set_ylabel(r'Counts')
#     # axs1[1].set_title('Final Std Energy Distribution per Bin')
#     axs1[1].set_yscale('log')

#     plt.tight_layout()
#     savefileName = f'HistogramStdMeanEnergy_{nProton}.pdf'
#     plt.savefig(savefileName)
#     plt.close(fig1) 

#     # Create 2D histogram plot for Angles
#     fig2, axs2 = plt.subplots(1, 2, figsize=(10, 6))
    
#     # Plot the histogram for the mean angle values
#     axs2[0].bar(angleEdges[:-1], meanAngleBin, width=np.diff(angleEdges), edgecolor="black", color='orange', align='edge')
#     axs2[0].set_xlabel(r'$E[\theta]$ (deg)')
#     axs2[0].set_ylabel(r'Counts')
#     # axs2[0].set_title('Final Mean Angle Distribution per Bin')
#     axs2[0].set_yscale('log')

#     # Plot the histogram for the standard deviation of angle
#     axs2[1].bar(angleEdges[:-1], stdAngleBin, width=np.diff(angleEdges), edgecolor="black", color='red', align='edge')
#     axs2[1].set_xlabel(r'$\sigma_\theta$ (deg)')
#     axs2[1].set_ylabel(r'Counts')
#     # axs2[1].set_title('Final Std Angle Distribution per Bin')
#     axs2[1].set_yscale('log')

#     plt.tight_layout()
#     savefileName = f'HistogramStdMeanAngle_{nProton}.pdf'
#     plt.savefig(savefileName)
#     plt.close(fig2)

    
if __name__ == "__main__":
    
    start_time = time.time()  # Record the start time
    
    # Variables
    numberOfProtonsRun = [1_000_000, 10_000]
    numberOfRuns = 20
    energy = 100
    
    numberOfBins = 100
        
    dataPath = '~/G4Data/'
    voxelPhaseFile = "./MyVoxelPhaseSpace.txt"
    fileName = "OutputVoxel.phsp"
    outputH5File = "SimulationMeanStd.h5"
    
    timeSimulation = []
    energyEdges = []
    angleEdges = []
    
    # Call the function to change input parameters
    modifyBeamEnergy(voxelPhaseFile, energy)
    
    # --- Step 1: Preliminary high-statistics pilot run to determine bin edges ---
    print("--- Phase 1: Determining optimal bin edges from a pilot run ---")
    pilotProtons = 1_000_000
    
    modifyInputParameters(voxelPhaseFile, pilotProtons)
    modifySeedRandomness(voxelPhaseFile, np.random.randint(0, 2**31 - 1))
    runTopas(voxelPhaseFile, dataPath)
    
    finalEnergyPilot, finalAnglesPilot = readInputFileSigma(fileName)
    
    # Find the true min and max values from the pilot run
    min_energy = min(finalEnergyPilot)
    max_energy = max(finalEnergyPilot)
    min_angle = min(finalAnglesPilot)
    max_angle = max(finalAnglesPilot)

    # Define the fixed bin edges using the true min/max values and the fixed bin count
    fixedEnergyEdges = np.linspace(min_energy, max_energy, numberOfBins + 1)
    fixedAngleEdges = np.linspace(min_angle, max_angle, numberOfBins + 1)
    
    print(f"Optimal energy range determined: {min_energy:.2f} to {max_energy:.2f} MeV.")
    print(f"Optimal angle range determined: {min_angle:.2f} to {max_angle:.2f} degrees.")
    print(f"Using {numberOfBins} bins for all subsequent runs.")
    
    # --- Step 2: Main statistical analysis for all specified proton counts ---
    print("\n--- Phase 2: Running simulations with fixed bin edges ---")
    
    for j, nProtons in enumerate(numberOfProtonsRun):
        modifyInputParameters(voxelPhaseFile, nProtons)
        totalEnergyBin = []
        totalAngleBin = []

        timeStart = time.time()

        # Process each run
        for i in range(numberOfRuns):
            modifySeedRandomness(voxelPhaseFile, i + nProtons * j)
            runTopas(voxelPhaseFile, dataPath)

            finalEnergy, finalAngles = readInputFileSigma(fileName)
            energyBin, energyEdges, angleBin, angleEdges = binNoise(finalEnergy, fixedEnergyEdges, finalAngles, fixedAngleEdges)
            
            totalEnergyBin.append(energyBin)
            totalAngleBin.append(angleBin)

        # Calculate mean and std for bins
        meanEnergyBin, stdEnergyBin, meanAngleBin, stdAngleBin = calculateMeanAndStd(totalEnergyBin, totalAngleBin)

        # Calculate total std as sqrt of summed variances
        total_variance = np.sum(stdEnergyBin**2)
        total_std = np.sqrt(total_variance)
        print("Sum of meanEnergyBin (average total counts):", np.sum(meanEnergyBin))
        print("Sum of stdEnergyBin (sum of fluctuations):", np.sum(stdEnergyBin))
        print("Total std (sqrt of summed variances):", total_std)

        timeEnd = time.time()
        timeSimulation.append(timeEnd - timeStart)

        # Plot 2D histogram
        plot2D(meanEnergyBin, stdEnergyBin, meanAngleBin, stdAngleBin, energyEdges, angleEdges, nProtons)
        
    # Plot time for each simulation
    plt.figure(figsize=(10, 8))
    plt.plot(numberOfProtonsRun, timeSimulation, linestyle='-',marker = '.', color="black", label=f"Energy {energy} MeV") # marker = '.' 
               
    plt.xlabel("Number of Histories")
    plt.ylabel("Time (seconds)")
    # plt.title(f"Energy {energy}")
    plt.legend()
        
    plt.savefig(f"./RunTimeNoise.pdf")
    # plt.show()
    plt.close()
    