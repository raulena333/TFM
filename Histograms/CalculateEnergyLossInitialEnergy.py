import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import os
import subprocess
import re

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
        
        
def readInputFileSigma(fileName):
    newData = np.loadtxt(fileName)  
    print(f'{fileName} loaded successfully.')
    finalEnergy, initialEnergy = newData[:, [5,10]].T
    normLoss = (initialEnergy - finalEnergy) / initialEnergy
    meanFinalEnergy = np.mean(finalEnergy)
    stdFinalEnergy = np.std(finalEnergy)
    
        
    return normLoss, meanFinalEnergy, stdFinalEnergy
           
if __name__ == "__main__":
    numberOfProtons = 100
    dataPath = '~/G4Data/'
    voxelPhaseFile = "./MyVoxelPhaseSpace.txt"
    fileName = './PhaseSpaceVoxel.phsp'
    
    energies = np.linspace(200, 9, 100)  # Energy values in MeV
    
    finalEnegies = {}
    meanEnergies = []
    stdEnergies = []

    # Call the function to change input parameters
    modifyInputParameters(voxelPhaseFile, numberOfProtons)

    for energy in energies:
        # Call the function to modify the beam energy
        modifyBeamEnergy(voxelPhaseFile, energy)  
        # Call the function to launch the simulation via terminal shell
        runTopas(voxelPhaseFile, dataPath)
        # Call the function to read the output file and extract energy and angles
        loss, mean, std = readInputFileSigma(fileName)
        finalEnegies[energy] = loss
        # Store the mean and std of final energies
        stdEnergies.append(std)
        meanEnergies.append(mean)
        
    # Save the results to a file
    np.save("normLossEnergies.npy",finalEnegies)