import time
import re
import subprocess
import numpy as np

def modifyNumberHistories(filePath, newHistories):
    """
    Modify the number of histories in run in a TOPAS input file and save it with the same name.

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - newEnergy (float): New energy value to replace in the file.
    """
    patternEnergy = r"(i:So/MySource/NumberOfHistoriesInRun = \d+"

    with open(filePath, 'r') as file:
        content = file.read()
    
    # Replace the energy value in the matched line
    updatedFile = re.sub(patternEnergy, rf"i:So/MySource/NumberOfHistoriesInRun = {newHistories}", content)
    
    with open(filePath, 'w') as file:
        file.write(updatedFile)

    print(f"Updated number of histories to {newHistories} in {filePath}.")
    
def runTopas(filePath, dataPath):
    """
    Run TOPAS txt-script with the modified input file

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - dataPath (str): Path to the TOPAS G4 data.
    """
    try:
        result = subprocess.run(f'export TOPAS_G4_DATA_DIR={dataPath} && ~raul/topas/bin/topas {filePath}', 
                        text=True, shell=True)
        if result == 0:
            print("Data loaded and simulation have started succesfully ")
            
    except FileNotFoundError:
        print("TOPAS executable not found. Make sure TOPAS is installed and in your PATH.")


if __name__ == "__main__":
    
    # Arguments
    dataPath = '~raul/G4Data/'
    voxelPhaseFile = "./MyVoxelPhaseSpace.txt"
    
    # Define number of protons launched
    nStart = 1000
    nEnd = 1000000
    step = 10
    numberOfHistories = np.arange(nStart, nEnd + 1, step)
    timeOfHistories = []
    
import time
import re
import subprocess
import numpy as np

def modifyNumberHistories(filePath, newHistories):
    """
    Modify the number of histories in a TOPAS input file and save it with the same name.

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - newHistories (int): New number of histories to replace in the file.
    """
    pattern = r"(i:So/MySource/NumberOfHistoriesInRun\s*=\s*)\d+"

    with open(filePath, 'r') as file:
        content = file.read()

    # Replace the number of histories in the matched line
    updatedFile = re.sub(pattern, rf"\1{newHistories}", content)

    with open(filePath, 'w') as file:
        file.write(updatedFile)

    print(f"Updated number of histories to {newHistories} in {filePath}.")

def runTopas(filePath, dataPath):
    """
    Run TOPAS with the modified input file.

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - dataPath (str): Path to the TOPAS G4 data directory.
    """
    try:
        command = f'export TOPAS_G4_DATA_DIR={dataPath} && ~raul/topas/bin/topas {filePath}'
        result = subprocess.run(command, text=True, shell=True, capture_output=True)

        if result.returncode == 0:
            print("Simulation started successfully.")
        else:
            print(f"Error running TOPAS:\n{result.stderr}")

    except FileNotFoundError:
        print("TOPAS executable not found. Make sure TOPAS is installed and in your PATH.")

if __name__ == "__main__":
    # Arguments
    dataPath = '~raul/G4Data/'
    voxelPhaseFile = "./MyVoxelPhaseSpace.txt"

    # Define number of protons launched
    nStart = 1000
    nEnd = 1000000
    step = 10
    numberOfHistories = np.arange(nStart, nEnd + 1, step)  # Include nEnd in range
    timeOfHistories = []

    # Start simulations
    for histories in numberOfHistories:
        modifyNumberHistories(voxelPhaseFile, histories)

        timeStart = time.time()
        runTopas(voxelPhaseFile, dataPath)
        timeEnd = time.time()

        time = timeEnd - timeStart
        timeOfHistories.append()
        
        # Calculate total elapsed time
        print(f"Process with {histories} histories took {timeEnd:.4f} seconds")

