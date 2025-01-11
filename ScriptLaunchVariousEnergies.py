import re
import subprocess

def modifyEnergyMaterial(filePath, newEnergy, newMaterial, newDensity):
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
    """
    try:
        updateData = subprocess.run([f'export TOPAS_G4_DATA_DIR=+{dataPath}'], shell = True)
        result = subprocess.run([f'~/topas/bin/topas {filePath}'], shell = True)
        if result == 0:
            print("Simulation haev started succesfully ")
            
    except FileNotFoundError:
        print("TOPAS executable not found. Make sure TOPAS is installed and in your PATH.")

if __name__ == "__main__":
    # Input parameters
    numberOfProtons = 50
    dataPath = '/home/Raul/G4Data/'
    voxelPhaseFile = "MyVoxelPhaseSpaceVR.txt"
    newEnergies = [110.0]
    newDensities = [1.0]
    newMaterials = ["Water"]

    # Create the dictionary for each material and density
    materials_dict = {material: {"density": density} for material, density in zip(newMaterials, newDensities)}
    print(materials_dict)
    #modifyInputParameters(voxelPhaseFile,numberOfProtons)
    
    for energy in newEnergies:   
        # Call the function to modify the beam energy
        modifyBeamEnergy(voxelPhaseFile, energy)  
        # Call the function to launch the simulation via terminal shell
        #runTopas(voxelPhaseFile, dataPath)
        print(f'Simulation for energy {energy} finished succesfully')
        

