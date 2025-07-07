import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import os
import re
import subprocess
import argparse

# Matplotlib params
params = {
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,      
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'legend.fontsize': 14
}
pylab.rcParams.update(params)  # Apply changes

def modifyMaterial(filePath, name, method):
    """
    Modify material of the subcomponents in a TOPAS input file.

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - name (str): Name of the material to set.
    """
    # Regular expression pattern to match the material assignment line
    if method == 'sheet':
        pattern = r'(s:Ge/myBox/Material = ")([^"]*)(")'
    elif method == 'sphere':
        pattern = r'(s:Ge/mySphere/Material = ")([^"]*)(")'
    
    with open(filePath, 'r') as file:
        content = file.read()
    
    # Replace only the material name inside the quotes
    updatedFile = re.sub(pattern, rf'\1{name}\3', content)

    # Write the updated content back to the file
    with open(filePath, 'w') as file:
        file.write(updatedFile)

    print(f"Updated material to '{name}' in {filePath}.")

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

def readData(fileName, totalNumberOfHistories):
    """
    Read data from a file and compute the fraction of particles that reached the voxel phase space.

    Parameters:
    - fileName (str): Path to the data file.
    - totalNumberOfHistories (int): Total number of simulated protons.

    Returns:
    - totalReachPhase (float): Fraction of protons reaching the voxel phase space,
                               or None if file is empty or unreadable.
    """
    try:
        newData = np.loadtxt(fileName)

        if newData.size == 0:
            print(f"{fileName} is empty.")
            return None

        # Ensure 2D shape for single-line files
        if newData.ndim == 1:
            newData = newData.reshape(1, -1)

        num_rows = newData.shape[0]
        totalReachPhase = num_rows / totalNumberOfHistories

        print(f"{fileName} loaded successfully: {num_rows} entries.")
        return totalReachPhase

    except Exception as e:
        print(f"Error reading {fileName}: {e}")
        return None
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="Select table threshold creation method (mutually exclusive). Example: --sheet or --sphere"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sheet', action='store_true', help="Create threshold tables for a sheet of 1 mm of thickness")
    group.add_argument('--sphere', action='store_true', help="Create threshold tables for a sphere of 1 mm of radius")
    
    return parser.parse_args()

if __name__ == "__main__":  
    args = parse_args()
    methodTable = 'sheet' if args.sheet else 'sphere'
    print(f"Selected method: {methodTable}")

    saveDir = "./Simulation/BraggPeak/"

    # Create directory if it doesn't exist
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    
    numberOfProtons = 1_000_000  # Number of protons to simulate
    dataPath = '~/G4Data/'
    
    baseFolder = {
        'sheet': {
            'voxelPhaseFile': './SheetVoxelTablesThreshold.txt',
            'fileName': './OutputVoxelSheetThreshold.phsp',
            'savePath': 'escapeProbsSheet.npz'
        },
        'sphere': {
            'voxelPhaseFile':  './SphereVoxelTablesThreshold.txt',
            'fileName': './OutputVoxelSphereThreshold.phsp',
            'savePath': 'escapeProbsSphere.npz'
        }
    }
    
    # Select the appropriate file paths based on the method
    voxelPhaseFile = baseFolder[methodTable]['voxelPhaseFile']
    fileName = baseFolder[methodTable]['fileName']
    saveFileName = baseFolder[methodTable]['savePath']
    
    # Dictionary: key = material index, value = Nx2 array: [energy, escape_prob]
    escapeProbs = {}
    
    materials = ['G4_LUNG_ICRP', 'G4_WATER', 'G4_BONE_CORTICAL_ICRP', 'G4_TISSUE_SOFT_ICRP']
    initialEnergies = [9.2, 9.0, 12.0, 9.2]
    finalEnergies = [8.9, 8.7, 11.6, 8.9]
    energyStep = 0.01  # decrement
    
    for materialIndex in range(len(initialEnergies)):
        initialEnergy = finalEnergies[materialIndex]
        finalEnergy = initialEnergies[materialIndex]
        modifyMaterial(voxelPhaseFile, materials[materialIndex], methodTable)
        energy = initialEnergy
        
        totalReachPhaseSpace = []
        energies = np.round(np.arange(initialEnergy, finalEnergy + energyStep / 2, energyStep), 2)

        for energy in energies:
            modifyInputParameters(voxelPhaseFile, numberOfProtons)
            modifyBeamEnergy(voxelPhaseFile, energy) 
            runTopas(voxelPhaseFile, dataPath)

            totalReachPhase = readData(fileName, numberOfProtons)
            if totalReachPhase is None:
                print(f"Stopping for material {materialIndex} at energy {energy:.2f} MeV")
                break

            totalReachPhaseSpace.append(totalReachPhase)
            
        # Save to dictionary as a NumPy array: [E, P] pairs
        escapeProbs[materialIndex] = np.array([energies, totalReachPhaseSpace]).T  
    
    escapeProbsStrkeys = {str(k): v for k, v in escapeProbs.items()}
    fullSavePath = os.path.join(saveDir, saveFileName)
    np.savez(fullSavePath, **escapeProbsStrkeys)
    print("Saved escapeProbs dictionary to escapeProbs.npz")
        
    # Load results and print them
    data = np.load(fullSavePath)
    for key, value in data.items():
        print(f"Material {key}: {value}")
    