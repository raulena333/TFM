import time
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

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
    

def modifyThreads(filePath, newThread):
    """
    Modify the number of threads in a TOPAS input file and save it with the same name.

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - newThread (float): New number of cores
    """
    patternEnergy = r"i:Ts/NumberOfThreads = \d+"

    with open(filePath, 'r') as file:
        content = file.read()
    
    # Replace the energy value in the matched line
    updatedFile = re.sub(patternEnergy, rf"i:Ts/NumberOfThreads = {newThread}", content)
    
    with open(filePath, 'w') as file:
        file.write(updatedFile)

    print(f"Updated number of cores to {newThread} in {filePath}.")
    
    
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
    voxelPhaseFile = "./MyVoxelPhaseSpace.txt"

    # Define number of protons launched and energies
    numberOfHistories = [100, 200, 300, 400, 500, 600, 700, 800, 900,
                        1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
                        10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 
                        100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 
                        1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000]
    energies = np.array([50., #100., 500., 1000.
                         ])
    cores = [2, 4, 6, 8, 10, 0]
    
    # Define colors for each energy level
    colors = ['b', 'g', 'r', 'm', 'c', 'y']  # Blue, Green, Red, Magenta, Cyan, Yellow  
    
    for i, energy in enumerate(energies):
        modifyBeamEnergy(voxelPhaseFile, energy)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Simulate depending on cores
        for j, core in enumerate(cores):
            modifyThreads(voxelPhaseFile, core)
            timeOfHistories = []
            
            # Start simulations
            for histories in numberOfHistories:
                modifyNumberHistories(voxelPhaseFile, histories)

                timeStart = time.time()
                runTopas(voxelPhaseFile, dataPath)
                timeEnd = time.time()

                timeOfSimulaton = timeEnd - timeStart
                timeOfHistories.append(timeOfSimulaton)
                
                # Calculate total elapsed time
                print(f"Process with {histories} histories took {timeEnd:.4f} seconds")

            # Plot
            plt.plot(numberOfHistories, timeOfHistories, linestyle='-', color=colors[j], label=f"Threads {core}") # marker = '.'    
        plt.xlabel("Number of Histories")
        plt.ylabel("Time (seconds)")
        plt.title(f"Energy {energy}")
        plt.legend()
        
        plt.savefig(f"fHistoryTimePlotForEnergies_Energy{energy}.pdf")
        # plt.show()
        plt.close()
    