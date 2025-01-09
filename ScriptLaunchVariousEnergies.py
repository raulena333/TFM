import re
import subprocess

def modifyBeamEnergy(filePath, newEnergy):
    """
    Modify the beam energy in a TOPAS input file and save it with the same name.

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - newEnergy (float): New energy value to replace in the file.
    """
    pattern = r"(dv:So/MySource/BeamEnergySpectrumValues = 1 )(\d+(\.\d+)?)( MeV)"
    with open(filePath, 'r') as file:
        content = file.read()
    
    # Replace the energy value in the matched line
    updatedFile = re.sub(pattern, rf"dv:So/MySource/BeamEnergySpectrumValues = 1 {newEnergy} MeV", content)
    
    with open(filePath, 'w') as file:
        file.write(updatedFile)

    print(f"Updated beam energy to {newEnergy} MeV in {filePath}.")

# 
def runTopas(filePath, dataPath):
    """
    Run TOPAS txt-script with the modified input file

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    """
    try:
        updateData = subprocess.run([f'export TOPAS_G4_DATA_DIR=+{dataPath}'], shell = True)
        result = subprocess.run(['~/topas/bin/topas', filePath], shell = True)
        if result == 0:
            print("Simulation haev started succesfully ")
            
    except FileNotFoundError:
        print("TOPAS executable not found. Make sure TOPAS is installed and in your PATH.")

if __name__ == "__main__":
    # Input parameters
    dataPath = '/home/Raul/G4Data/'
    voxelPhaseFile = "MyVoxelPhaseSpaceVR.txt"
    newEnergies = [120.0, 150.0, 300.0]
    
    for energy in newEnergies:   
        # Call the function to modify the beam energy
        modifyBeamEnergy(voxelPhaseFile, energy)  
        # Call the function to launch the simulation via terminal shell
        runTopas(voxelPhaseFile, dataPath)
        print(f'Simulation for energy {energy} finished succesfully')
        

