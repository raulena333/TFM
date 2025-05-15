import time
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
import csv

# Plot appearance settings
params = {
    'xtick.labelsize': 17,    
    'ytick.labelsize': 17,      
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'legend.fontsize': 16
}
pylab.rcParams.update(params) 

def modifyNumberHistories(filePath, newHistories):
    """
    Modify the number of histories in run in a TOPAS input file and save it with the same name.

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - newEnergy (float): New energy value to replace in the file.
    """
    patternEnergy = r"i:So/MySource/NumberOfHistoriesInRun = \d+"

    with open(filePath, 'r') as file:
        content = file.read()
    
    # Replace the energy value in the matched line
    updatedFile = re.sub(patternEnergy, rf"i:So/MySource/NumberOfHistoriesInRun = {newHistories}", content)
    
    with open(filePath, 'w') as file:
        file.write(updatedFile)

    print(f"Updated number of histories to {newHistories} in {filePath}.")
      
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
    
def runTopas(filePath, dataPath):
    """
    Run TOPAS txt-script with the modified input file

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - dataPath (str): Path to the TOPAS G4 data.
    """
    try:
        result = subprocess.run(f"export TOPAS_G4_DATA_DIR={dataPath} && ~/topas/bin/topas {filePath}", 
               text=True, shell=True)
        if result == 0:
            print("Data loaded and simulation have started succesfully ")
            
    except FileNotFoundError:
        print("TOPAS executable not found. Make sure TOPAS is installed and in your PATH.")
        

if __name__ == "__main__":
    # Arguments
    dataPath = '~/G4Data/'
    voxelPhaseFile = "./MyVoxelPhaseSpaceBigCube.txt"
    timingPath = './TimingTOPAS/'
    
    if not os.path.exists(timingPath):
        os.makedirs(timingPath)

    # Define number of protons launched and energies
    numberOfHistories = [1000000]
    energy = 200
    modifyBeamEnergy(voxelPhaseFile, energy)
    
    timeOfHistories = []
    
    for numberRuns in numberOfHistories:
        modifyNumberHistories(voxelPhaseFile, numberRuns)

        timeStart = time.time()
        runTopas(voxelPhaseFile, dataPath)
        timeEnd = time.time()

        timeOfSimulaton = timeEnd - timeStart
        timeOfHistories.append(timeOfSimulaton)
                
        # Calculate total elapsed time
        print(f"Process with {numberRuns} histories took {timeOfSimulaton:.4f} seconds")
        
    # Save time of simulation in CSV
    with open(timingPath + f"TimeOfSimulationBigVoxelTOPAS.csv", 'w') as file:
        file.write(
                    "# Number of histories; Time in seconds\n"
                )
        writer = csv.writer(file, delimiter=' ')
        for histories, tIme in zip(numberOfHistories, timeOfHistories):
            writer.writerow([histories, tIme])