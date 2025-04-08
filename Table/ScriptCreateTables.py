import re
import os
import subprocess
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

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


def modifyMaterial(filePath, name):
    """
    Modify material of the subcomponents in a TOPAS input file.

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - name (str): Name of the material to set.
    """
    # Regular expression pattern to match the material assignment line
    pattern = r'(s:Ge/myBox/Material = ")([^"]*)(")'
    
    with open(filePath, 'r') as file:
        content = file.read()
    
    # Replace only the material name inside the quotes
    updatedFile = re.sub(pattern, rf'\1{name}\3', content)

    # Write the updated content back to the file
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
        
        
def calculateAngleEnergyProbabilities(fileName,  numberOfBins):
    """
    Calculate and return the angles and final energies for an initial enrgy

    Parameter:
    - filePath (str): Path to the TOPAS output file.
    """
    # Define the threshold for energy and angle
    threshold = [0, -0.57, 70] # [maxEnergy, minEnergy, angleThreshold]
    
    # Load files
    newData = np.loadtxt(fileName)  
    print(f'{fileName} loaded successfully.')
    finalDirectionCosineX, finalDirectionCosineY, finalEnergy, isSign, initialEnergy = newData[:, [3,4,5,8,10]].T
    logFinalEnergy = np.log((initialEnergy - finalEnergy) / initialEnergy)
    
    # Calculate variable change for energy
    logFinalEnergy *= 1 / np.sqrt(initialEnergy)
    
    # Apply threshold to filter data energies
    uniformMaxEnergy = threshold[0]
    uniformMinEnergy = threshold[1]
    uniformAngleThreshold = threshold[2]
            
    mask = (logFinalEnergy < uniformMaxEnergy) & (logFinalEnergy > uniformMinEnergy)
    filteredFinalDirectionCosineX = finalDirectionCosineX[mask]
    filteredFinalDirectionCosineY = finalDirectionCosineY[mask]
    filteredIsSign = isSign[mask]
    logFinalEnergy = logFinalEnergy[mask]
    energy = np.mean(initialEnergy)
    
    # Calculate the angles and apply the energy scaling and threshold
    finalAngles = []
    indexToDelete = []

    # Loop over each particle's direction cosines
    for j, (directionX, directionY, sign) in enumerate(zip(filteredFinalDirectionCosineX, filteredFinalDirectionCosineY, filteredIsSign)):
        directionZ = np.sqrt(1 - directionX**2 - directionY**2)
            
        # Adjust sign of directionZ based on isSign
        if sign == 0:
            directionZ *= -1
        angle = np.degrees(np.arccos(directionZ))
        
        # Apply energy scaling to the angle
        angle *= np.sqrt(energy)
            
        if angle > uniformAngleThreshold:
            indexToDelete.append(j)
        else:
            finalAngles.append(angle)
            
    if indexToDelete:
        indexToDelete = np.array(indexToDelete, dtype=int)
        logFinalEnergy = np.delete(logFinalEnergy, indexToDelete)
        
    # Compute 2D Histogram
    hist1, _ , _ = np.histogram2d(finalAngles, logFinalEnergy, bins=numberOfBins,  range = ([0, uniformAngleThreshold], [uniformMinEnergy, uniformMaxEnergy]))
    finalProbabilities = hist1 / np.sum(hist1)
        
    return finalProbabilities

# https://numpy.org/doc/2.2/reference/generated/numpy.savez.html
def saveToNPZ(probabilityTable, materialList, energyList, savePath):
    """
    Save the final 4D probability tables in a compressed NPZ file for CUDA processing.

    Parameters:
    - probabilityTable (dict): Nested dict: material -> energy -> 2D numpy array
    - materialList (list): List of material names
    - energyList (list): List of energy values (float)
    - savePath (str): Path to save the .npz file
    """
    num_materials = len(materialList)
    num_energies = len(energyList)
    
    os.makedirs(os.path.dirname(savePath), exist_ok=True)  # Create directory if it doesn't exist

    # Get shape of 2D hist
    angle_bins, energy_bins = probabilityTable[materialList[0]][energyList[0]].shape

    # Create 4D array
    prob_array = np.zeros((num_materials, num_energies, angle_bins, energy_bins), dtype=np.float32)

    for i, mat in enumerate(materialList):
        for j, en in enumerate(energyList):
            prob_array[i, j] = probabilityTable[mat][en]

    np.savez_compressed(
        savePath,
        prob_table=prob_array,
        materials=np.array(materialList),
        energies=np.array(energyList, dtype=np.float32)
    )
    print(f"Saved efficient 4D CUDA table to {savePath}")

    
if __name__ == "__main__":  
    # Variables    
    numberOfProtons = [100, 1000, 10000, 100000 # 1000000, 10000000
                       ] # Number of protons to simulate
    dataPath = '~/G4Data/'
    voxelPhaseFile = './MyVoxelPhaseSpace.txt'
    fileName = 'OutputVoxel.phsp'
    
    # Input parameters
    initialEnergy = 200.
    finalEnergy = 15.
    stepEnergy = 0.1
    energies = np.round(np.arange(initialEnergy, finalEnergy - stepEnergy, -stepEnergy), 1)
      
    materials = ['G4_LUNG_ICRP', 'G4_WATER', 'G4_AIR', 'G4_BONE_CORTICAL_ICRP', 'G4_TISSUE_SOFT_ICRP' ] # Materials to simulate
    densities = [1.0, 0.00120479, 1.92, 1.04, 1.03] # g/cm^3

    timeSimulation = []
    
    for nProtons in numberOfProtons:
        
        savePath = f'./Table/4DTable{nProtons}.npz'
        
        modifyInputParameters(voxelPhaseFile, nProtons)     
        resultsTable = {}  # Dictionary to store all results before saving to .npz   

        startTime = time.time()
        
        for material in materials:

            # Modify the material in the TOPAS file
            modifyMaterial(voxelPhaseFile, material)
            
            resultsTable[material] = {}
            
            for energy in energies:  
                # Initialize lists for energy and angle values
                energyVec = []
                angleVec = []
                
                # Modify beam energy for the current run
                modifyBeamEnergy(voxelPhaseFile, energy)  
                # Run TOPAS simulation
                runTopas(voxelPhaseFile, dataPath)
                # Retrieve energy and angle distributions probabilities
                finalProbabilities = calculateAngleEnergyProbabilities(fileName, numberOfBins=100)
                
                # Store in dictionary for NPZ saving   
                resultsTable[material][energy] = finalProbabilities
                            
        # Save to CUDA-optimized NPZ
        saveToNPZ(resultsTable, materials, energies, savePath)
        
        timeEnd = time.time()
        elapsedTime = timeEnd - startTime
        timeSimulation.append(elapsedTime)
        
    # Plot time for each table creation
    plt.figure(figsize=(10, 8))
    plt.plot(numberOfProtons, timeSimulation, linestyle='-',marker = '.', color="black") 
               
    plt.xlabel("Number of Histories")
    plt.ylabel("Time (seconds)")
        
    plt.savefig(f"./RunTimeCreateTables.pdf")
    # plt.show()
    plt.close()