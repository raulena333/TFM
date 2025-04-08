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
    'xtick.labelsize': 12,    
    'ytick.labelsize': 12,      
    'axes.titlesize' : 12,
    'axes.labelsize' : 12,
    'legend.fontsize': 12,
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


def binNoise(energies, numberOfBinsEnergies, angles, numberOfBinsAngles, nProtons, nRun):
    # Compute histogram counts for energies
    energyCounts, energyEdges = np.histogram(energies, bins=numberOfBinsEnergies)

    # Compute histogram counts for angles
    angleCounts, angleEdges = np.histogram(angles, bins=numberOfBinsAngles)
    
    filePath = "./Plots_{nProtons}/".format(nProtons=nProtons)
    if not os.path.exists(filePath):
        os.makedirs(filePath, exist_ok=True)
        
    # Create 2Dhistogram plot
    fig1, axs1 = plt.subplots(1, 2, figsize=(10, 6))

    sns.histplot(energies, bins=numberOfBinsEnergies, edgecolor="black", color='orange', kde=False, ax=axs1[0])
    axs1[0].set_xlabel(r'$E[E_f]$ (MeV)')
    axs1[0].set_title('Final Energy Distribution')
    axs1[0].set_yscale('log')
            
    sns.histplot(angles, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False, ax=axs1[1])
    axs1[1].set_xlabel(r'$E[\theta]$ (deg)')
    axs1[1].set_title('Angle Distribution')
    axs1[1].set_yscale('log')

    plt.tight_layout()
    # plt.show()
    savefileName = f'{filePath}HistogramStdMeanEnergyRun_{nRun}.pdf'
    plt.savefig(savefileName)
    plt.close(fig1) 
     
    return energyCounts, energyEdges, angleCounts, angleEdges

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
    # Create 2D histogram plot for Energy
    fig1, axs1 = plt.subplots(1, 2, figsize=(10, 6))

    # Plot the histogram for the mean energy values
    axs1[0].bar(energyEdges[:-1], meanEnergyBin, width=np.diff(energyEdges), edgecolor="black", color='orange', align='edge')
    axs1[0].set_xlabel(r'$E[E_f]$ (MeV)')
    axs1[0].set_ylabel(r'Counts')
    axs1[0].set_title('Final Mean Energy Distribution per Bin')
    axs1[0].set_yscale('log')

    # Plot the histogram for the standard deviation of energy
    axs1[1].bar(energyEdges[:-1], stdEnergyBin, width=np.diff(energyEdges), edgecolor="black", color='red', align='edge')
    axs1[1].set_xlabel(r'$\sigma_{E_f}$ (MeV)')
    axs1[1].set_ylabel(r'Counts')
    axs1[1].set_title('Final Std Energy Distribution per Bin')
    axs1[1].set_yscale('log')

    plt.tight_layout()
    savefileName = f'HistogramStdMeanEnergy_{nProton}.pdf'
    plt.savefig(savefileName)
    plt.close(fig1) 

    # Create 2D histogram plot for Angles
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 6))
    
    # Plot the histogram for the mean angle values
    axs2[0].bar(angleEdges[:-1], meanAngleBin, width=np.diff(angleEdges), edgecolor="black", color='orange', align='edge')
    axs2[0].set_xlabel(r'$E[\theta]$ (deg)')
    axs2[0].set_ylabel(r'Counts')
    axs2[0].set_title('Final Mean Angle Distribution per Bin')
    axs2[0].set_yscale('log')

    # Plot the histogram for the standard deviation of angle
    axs2[1].bar(angleEdges[:-1], stdAngleBin, width=np.diff(angleEdges), edgecolor="black", color='red', align='edge')
    axs2[1].set_xlabel(r'$\sigma_\theta$ (deg)')
    axs2[1].set_ylabel(r'Counts')
    axs2[1].set_title('Final Std Angle Distribution per Bin')
    axs2[1].set_yscale('log')

    plt.tight_layout()
    savefileName = f'HistogramStdMeanAngle_{nProton}.pdf'
    plt.savefig(savefileName)
    plt.close(fig2)

    
if __name__ == "__main__":
    
    start_time = time.time()  # Record the start time
    
    # Variables
    numberOfProtonsRun = [1000, 100000, 1000000, 10000000]
    numberOfRuns = 20
    energy = 100
    
    numberOfBinsAngles = 10
    numberOfBinsEnergies = 10
    jointNumberOfBins = 10
    
    dataPath = '~/G4Data/'
    voxelPhaseFile = "./MyVoxelPhaseSpace.txt"
    fileName = "OutputVoxel.phsp"
    outputH5File = "SimulationMeanStd.h5"
    
    timeSimulation = []
    energyEdges = []
    angleEdges = []
            
    # Call the function to change input parameters
    modifyBeamEnergy(voxelPhaseFile, energy)
    
    for j, nProtons in enumerate(numberOfProtonsRun):
        modifyInputParameters(voxelPhaseFile, nProtons)
        totalEnergyBin = []
        totalAngleBin = []

        timeStart = time.time()

        # Process each run
        for i in range(numberOfRuns):
            modifySeedRandomness(voxelPhaseFile, i + nProtons)
            runTopas(voxelPhaseFile, dataPath)

            finalEnergy, finalAngles = readInputFileSigma(fileName)
            energyBin, energyEdges, angleBin, angleEdges = binNoise(finalEnergy, numberOfBinsEnergies, finalAngles, numberOfBinsAngles, nProtons, i)
            
            totalEnergyBin.append(energyBin)
            totalAngleBin.append(angleBin)

        # Calculate mean and std for bins
        meanEnergyBin, stdEnergyBin, meanAngleBin, stdAngleBin = calculateMeanAndStd(totalEnergyBin, totalAngleBin)

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
    