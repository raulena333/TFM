import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import os
import re
import subprocess

# Matplotlib params
params = {
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,      
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'legend.fontsize': 14
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

def readData(fileName):
    """
    Read data from a file and extract relevant information.

    Parameters: 
    - fileName (str): Path to the data file.

    Returns:
    - logFinalEnergy (array), finalAngles (list), totalReachPhase (float)
      or (None, None, None) if data is invalid or missing
    """
    try:
        newData = np.loadtxt(fileName)

        # Handle empty files or invalid content
        if newData.size == 0:
            print(f"{fileName} is empty.")
            return None, None, None

        print(f'{fileName} loaded successfully.')

        try:
            finalDirectionCosineX, finalDirectionCosineY, finalEnergy, isSign, initialEnergy = newData[:, [3, 4, 5, 8, 10]].T
        except Exception as e:
            print(f"Unpacking error: {e}")
            return None, None, None

        logFinalEnergy = np.log((initialEnergy - finalEnergy) / initialEnergy) / np.sqrt(initialEnergy)
        totalReachPhase = len(logFinalEnergy) / numberOfProtons * 100
        energy = np.mean(initialEnergy)

        # Compute final angles
        finalAngles = []
        for directionX, directionY, sign in zip(finalDirectionCosineX, finalDirectionCosineY, isSign):
            value = 1 - directionX**2 - directionY**2
            value = np.maximum(value, 0)
            directionZ = np.sqrt(value)
            if sign == 0:
                directionZ *= -1
            angle = np.degrees(np.arccos(directionZ)) * np.sqrt(energy)
            finalAngles.append(angle)

        return logFinalEnergy, finalAngles, totalReachPhase

    except Exception as e:
        print(f"Error reading {fileName}: {e}")
        return None, None, None
    
           
if __name__ == "__main__":

    savePath = "./Plots/"

    # Create directory if it doesn't exist
    if not os.path.exists(savePath):
        os.makedirs(savePath)
        
    # Number of bins for histograms
    numberOfBinsAngles = 100
    numberOfBinsEnergies = 100
    jointNumberOfBins = 100
    
    numberOfProtons = 100000
    dataPath = '~/G4Data/'
    voxelPhaseFile = "./MyVoxelPhaseSpaceLowEnergy.txt"
    fileName = f"./LowerEnergyVoxel.phsp"
    
    totalNumberReachPhase = []
    initialEnergy = 10 # MeV
    energyStep = -0.1  # decrement
    energy = initialEnergy
    energies = []

    while energy > 0:
        modifyInputParameters(voxelPhaseFile, numberOfProtons)
        modifyBeamEnergy(voxelPhaseFile, energy) 
         
        # Call the function to launch the simulation via terminal shell
        runTopas(voxelPhaseFile, dataPath)
        
        # Read data
        logFinalEnergy, finalAngles, totalReachPhase = readData(fileName)
        
        # Stop condition: if readData returned None values
        if logFinalEnergy is None or finalAngles is None or totalReachPhase is None:
            print(">>> Data could not be read — stopping simulation.")
            break

        totalNumberReachPhase.append(totalReachPhase)
        energies.append(energy)
        energy += energyStep 
                
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
        plt.savefig(f'{savePath}BigVoxelHistograms{energy:.2f}.pdf')
        plt.close(fig)

        
        hist1, xedges1, yedges1 = np.histogram2d(finalAngles, logFinalEnergy, bins=jointNumberOfBins)
        finalProbabilities = hist1 / np.sum(hist1)

        fig2, axs2 = plt.subplots(figsize=(8, 6))
        h1 = axs2.pcolormesh(xedges1, yedges1, finalProbabilities.T, cmap='Reds', shading='auto')
        fig2.colorbar(h1, ax=axs2, label='Probability')
        axs2.set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
        axs2.set_ylabel(r'$ln((E_i-E_f)/E_i)\sqrt{E_i}$ (MeV$^{-1/2}$)')

        plt.tight_layout()
        plt.savefig(f'{savePath}2DBigVoxelHistograms{energy:.2f}.pdf')
        plt.close(fig2) 
        
    
    figure = plt.figure(figsize=(8, 6))
    plt.scatter(energies, totalNumberReachPhase, color='black', s = 20, marker = '.')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Total Reach Phase (%)')
    plt.tight_layout()
    plt.savefig(f'./TotalReachPhase.pdf')
    plt.close(figure)
    