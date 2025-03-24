import re
import subprocess
import argparse

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
        result = subprocess.run(f'export TOPAS_G4_DATA_DIR={dataPath} && ~raul/topas/bin/topas {filePath}', 
                        text=True, shell=True)
        if result == 0:
            print("Data loaded and simulation have started succesfully ")
            
    except FileNotFoundError:
        print("TOPAS executable not found. Make sure TOPAS is installed and in your PATH.")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Choose simulation mode.")
    parser.add_argument("--mode", choices=["one", "various"], default="various", help="Select the mode to run the script.")
    args = parser.parse_args()

    numberOfProtons = 1000
    dataPath = '~raul/G4Data/'
    voxelPhaseFile = "./MyVoxelPhaseSpace.txt"

    if args.mode == "one":
        # Input parameters
        energies = [150, 125, 100, 75, 50]

        # Call the function to change input parameters
        modifyInputParameters(voxelPhaseFile, numberOfProtons)

        for energy in energies:
            # Call the function to modify the beam energy
            modifyBeamEnergy(voxelPhaseFile, energy)  
            # Call the function to launch the simulation via terminal shell
            runTopas(voxelPhaseFile, dataPath)
            print(f'Simulation for energy {energy:.2f} finished successfully')
    else:
        # Input parameters
        initialEnergy = 110.0
        finalEnergy = 90
        stepEnergy = 0.5

        # Call the function to change input parameters
        modifyInputParameters(voxelPhaseFile, numberOfProtons)

        currentEnergy = initialEnergy
        while currentEnergy >= finalEnergy:   
            # Call the function to modify the beam energy
            modifyBeamEnergy(voxelPhaseFile, currentEnergy)  
            # Call the function to launch the simulation via terminal shell
            runTopas(voxelPhaseFile, dataPath)
            print(f'Simulation for energy {currentEnergy:.2f} finished successfully')
            currentEnergy -= stepEnergy 
