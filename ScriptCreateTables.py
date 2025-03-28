import re
import subprocess
import numpy as np
import argparse
import h5py

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
    
def modifyMaterial(filePath, name):
    """
    Modify material of the subcomponents in a TOPAS input file.

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - name (str): Name of the material to set.
    """

    for j in range(1, 3):  # Iterate over subcomponents 1 and 2
        with open(filePath, 'r') as file:
            content = file.read()
        
        # Regular expression pattern to match the material assignment line
        pattern = rf'(s:Ge/subComponent{j}/Material\s*=\s*")[^"]*(")'
        
        # Replace only the material name inside the quotes
        updatedFile = re.sub(pattern, rf'\1{name}\2', content)

        with open(filePath, 'w') as file:
            file.write(updatedFile)

    print(f"Updated material to '{name}' in {filePath}.")

    
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
        
        
def calculateAngleEnergy(fileName):
    """
    Calculate and return the angles and final energies for an initial enrgy

    Parameter:
    - filePath (str): Path to the TOPAS output file.
    """
    
    # Load files
    newData = np.loadtxt(fileName)  
    print(f'{fileName} loaded successfully.')
    finalDirectionCosineX, finalDirectionCosineY, finalEnergy, isSign, initialEnergy, initialDirectionCosineZ = newData[:, [3,4,5,8]].T
    
    finalAngles = []
        
    # Loop over each particle's direction cosines
    for directionX, directionY, sign in zip(finalDirectionCosineX, finalDirectionCosineY, isSign):
        directionZ = np.sqrt(1 - directionX**2 - directionY**2)
            
        # Adjust sign of directionZ based on isSign
        if sign == 0:
            directionZ *= -1
        angle = np.degrees(np.arccos(directionZ))
            
        finalAngles.append(angle)
    
    return finalAngles, finalEnergy

# https://docs.h5py.org/en/stable/high/dataset.html#creating-datasets
def saveTohdf5(material, energy, energyVec, angleVec):
    """
    Create and save a h5 file for each material and initial energy

    Parameters:
    - material (str): Material of the voxel.
    - energy (float): Initial energy of the proton.
    - energyVec (array): Array of the dispersed angle.
    - angleVec (array): Array of the final energies of the protons.
    """
    
    with h5py.File('4DTableEnergy.h5', 'a') as file:
        group = file.require_group(f"{material}/{energy}")  # Create or open group
        
        # Save with compression
        group.create_dataset("EnergyVec", data=np.array(energyVec), compression="gzip")
        group.create_dataset("AngleVec", data=np.array(angleVec), compression="gzip")

           
if __name__ == "__main__":  
    # Variables    
    numberOfProtons = 10000
    dataPath = '~/G4Data/'
    voxelPhaseFile = "./MyVoxelPhaseSpace.txt"
    fileName = "OutputVoxel.phsp"
    
    # Input parameters
    initialEnergy = 110.0
    finalEnergy = 90
    stepEnergy = 0.3
        
    materials = ['G4_WATER']
    densities = ['1.0'] # g/cm^3
        
    # 4D Table: Dictionary to store results
    resultsTable = {}  

    for material in materials:
        resultsTable[material] = {} 

        # Modify the material in the TOPAS file
        modifyMaterial(voxelPhaseFile, material)

        currentEnergy = initialEnergy
        while currentEnergy >= finalEnergy:  
            # Initialize lists for energy and angle values
            energyVec = []
            angleVec = []
            
            # Modify beam energy for the current run
            modifyBeamEnergy(voxelPhaseFile, currentEnergy)  
            # Run TOPAS simulation
            runTopas(voxelPhaseFile, dataPath)
            # Retrieve energy and angle distributions
            angleVec, energyVec = calculateAngleEnergy(fileName)
            # Save the Table 
            saveTohdf5(material,currentEnergy, energyVec, angleVec)
            
            currentEnergy -= stepEnergy 



