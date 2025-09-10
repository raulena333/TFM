import re
import os
import subprocess
import numpy as np
import argparse
import time

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
        

def calculateAngleEnergyProbabilities(fileName, numberOfBins):
    """
    Calculate and return 2D histograms of angle vs. energy for both transformed and normalized methods.

    Parameters:
    - fileName (str): Path to the TOPAS output file.
    - material (str): Material name for threshold selection.
    - numberOfBins (int): Number of bins for 2D histogram in both angle and energy.
    
    Returns:
    - finalProbabilitiesTrans (ndarray): 2D histogram (angle vs transformed log energy).
    - finalProbabilitiesNorm (ndarray): 2D histogram (normalized angle vs. normalized energy).
    - maxTheta (float): Maximum angle found (degrees).
    - finalEnergyMin (float): Minimum final energy for normalization.
    - finalEnergyMax (float): Maximum final energy for normalization.
    """
    # Thresholds for transformed energy and angular filtering
    threshold = [0, -0.6, 70]  # [maxLogE, minLogE, maxAngleDeg]
    
    # Load data
    data = np.loadtxt(fileName)
    print(f'{fileName} loaded successfully.')

    finalDirX, finalDirY, finalEnergy, isSign, initialEnergy = data[:, [3, 4, 5, 8, 10]].T
    energy = np.mean(initialEnergy)
 
    # ---------- TRANSFORMED ENERGY PATH ----------
    logE = np.log((initialEnergy - finalEnergy) / initialEnergy)
    logE *= 1 / np.sqrt(initialEnergy)
    
    # Clamp logE to the valid range
    logE = np.clip(logE, threshold[1], threshold[0])

    maskTrans = (logE < threshold[0]) & (logE > threshold[1])
    logE_T = logE[maskTrans]
    dirX_T = finalDirX[maskTrans]
    dirY_T = finalDirY[maskTrans]
    sign_T = isSign[maskTrans]
    finalEnergy = finalEnergy[maskTrans]

    dirZ_T = np.sqrt(np.clip(1 - dirX_T**2 - dirY_T**2, 0, 1))
    dirZ_T[sign_T == 0] *= -1
    angle = np.degrees(np.arccos(np.clip(dirZ_T, -1.0, 1.0))) 
    angle_T = angle * np.sqrt(energy)
    
    # Clam angle to the valid range
    angle_T = np.clip(angle_T, 0, threshold[2])

    # Apply angular cutoff
    angle_mask = angle_T <= threshold[2]
    angle = angle[angle_mask]
    angle_T = angle_T[angle_mask]
    logE_T = logE_T[angle_mask]
    finalEnergy = finalEnergy[angle_mask]
    
    # ---------- NORMALIZED ENERGY PATH ----------
    maxTheta = np.max(angle)
    minTheta = np.min(angle)
    angle_N_norm = (angle - minTheta) / (maxTheta - minTheta)

    # Normalize final energy for histogram
    finalEnergyMin = np.min(finalEnergy)
    finalEnergyMax = np.max(finalEnergy)
    energy_Norm = (finalEnergy - finalEnergyMin) / (finalEnergyMax - finalEnergyMin)

    # ---------- HISTOGRAMS ----------
    histTrans, _, _ = np.histogram2d(angle_T, logE_T,
                                     bins=numberOfBins,
                                     range=([0, threshold[2]], [threshold[1], threshold[0]]))
    finalProbabilitiesTrans = histTrans / np.sum(histTrans)

    histNorm, _, _ = np.histogram2d(angle_N_norm, energy_Norm,
                                    bins=numberOfBins,
                                    range=([0, 1], [0, 1]))
    finalProbabilitiesNorm = histNorm / np.sum(histNorm)

    return finalProbabilitiesTrans, finalProbabilitiesNorm, maxTheta, minTheta, finalEnergyMin, finalEnergyMax


# https://numpy.org/doc/2.2/reference/generated/numpy.savez.html
def saveToNPZ(probabilityTableTrans, probabilityTableNorm, materialList, energyList, thetaMaxArrayDict, thetaMinArrayDict, 
            energyMinArrayDict, energyMaxArrayDict, savePathTrans, savePathNorm):
    """
    Save the final 4D probability tables in a compressed NPZ file for CUDA processing.

    Parameters:
    - probabilityTableTrans (dict): Dictionary of 2D probability tables for each material and energy
    - probabilityTableNorm (dict): Dictionary of 2D probability tables for each material and energy
    - materialList (list): List of material names
    - energyList (list): List of energy values (float)
    - thetaMaxArrayDict (dict): Dictionary of maximum theta values for each material and energy
    - thetaMinArrayDict (dict): Dictionary of minimum theta values for each material and energy
    - energyMinArrayDict (dict): Dictionary of minimum energy values for each material and energy
    - energyMaxArrayDict (dict): Dictionary of maximum energy values for each material and energy
    - savePathTrans (str): Path to save the compressed NPZ file
    - savePathNorm (str): Path to save the compressed NPZ file 
    """
    numMaterials = len(materialList)
    numEnergies = len(energyList)
    
    os.makedirs(os.path.dirname(savePathTrans), exist_ok=True)
    os.makedirs(os.path.dirname(savePathNorm), exist_ok=True)
    
    # Get shape of 2D hist
    angleBinsTrans, energyBinsTrans = probabilityTableTrans[materialList[0]][energyList[0]].shape
    angleBinsNorm, energyBinsNorm = probabilityTableNorm[materialList[0]][energyList[0]].shape

    # Create 4D array
    probArrayTrans = np.zeros((numMaterials, numEnergies, angleBinsTrans, energyBinsTrans), dtype=np.float32)
    probArrayNorm = np.zeros((numMaterials, numEnergies, angleBinsNorm, energyBinsNorm), dtype=np.float32)

    # Create 2D arrays for thetaMax and finalEnergyMin 
    thetaMaxArray = np.zeros((numMaterials, numEnergies), dtype=np.float32)
    thetaMinArray = np.zeros((numMaterials, numEnergies), dtype=np.float32)
    energyMinArray = np.zeros((numMaterials, numEnergies), dtype=np.float32)
    energyMaxArray = np.zeros((numMaterials, numEnergies), dtype=np.float32)

    for i, mat in enumerate(materialList):
        for j, en in enumerate(energyList):
            probArrayNorm[i, j] = probabilityTableNorm[mat][en]
            thetaMaxArray[i, j] = thetaMaxArrayDict[mat][en]
            thetaMinArray[i, j] = thetaMinArrayDict[mat][en]
            energyMinArray[i, j] = energyMinArrayDict[mat][en]
            energyMaxArray[i, j] = energyMaxArrayDict[mat][en]
            
            probArrayTrans[i, j] = probabilityTableTrans[mat][en]
         
    np.savez_compressed(
        savePathNorm,
        probTable= probArrayNorm,
        thetaMax= thetaMaxArray,
        thetaMin= thetaMinArray,
        energyMin= energyMinArray,
        energyMax= energyMaxArray,
        materials= np.array(materialList),
        energies= np.array(energyList, dtype=np.float32)
    )
    print(f"Saved efficient 4D CUDA table to {savePathNorm}")
    
    np.savez_compressed(
        savePathTrans,
        probTable=probArrayTrans,
        materials=np.array(materialList),
        energies=np.array(energyList, dtype=np.float32)
    )
    print(f"Saved efficient 4D CUDA table to {savePathTrans}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Select table creation method (mutually exclusive). Example: --sheet or --sphere"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sheet', action='store_true', help="Create tables for a sheet of 1 mm of thickness")
    group.add_argument('--sphere', action='store_true', help="Create tables for a sphere of 1 mm of radius")
    
    return parser.parse_args()

if __name__ == "__main__":  
    
    args = parse_args()
    methodTable = 'sheet' if args.sheet else 'sphere'
    print(f"Selected method: {methodTable}")

    # Variables    
    numberOfProtons = 100_000 # Number of protons to simulate
    dataPath = '~/G4Data/'
    
    baseFolder = {
        'sheet': {
            'voxelPhaseFile': './SheetVoxelTables.txt',
            'fileName': './OutputVoxelSheet.phsp',
            'savePathTrans': './Table/4DTableTransSheet.npz',
            'savePathNorm': './Table/4DTableNormSheet.npz'
        },
        'sphere': {
            'voxelPhaseFile':  './SphereVoxelTables.txt',
            'fileName': './OutputVoxelSphere.phsp',
            'savePathTrans': './Table/4DTableTransSphere.npz',
            'savePathNorm': './Table/4DTableNormSphere.npz'
        }
    }
    
    # Input parameters
    numberOfBins = 100
    
    # First segment: 200.6 MeV to 25 MeV with 0.2 MeV steps
    highSegment = np.round(np.arange(200.6, 25.0, -0.2), 1)
    # Second segment: 25 MeV to 8.7 MeV with 0.1 MeV steps
    lowSegment = np.round(np.arange(25.0, 8.7 - 0.01, -0.1), 1) 
    # Concatenate both segments
    energies = np.concatenate((highSegment, lowSegment))

    materials = ['G4_LUNG_ICRP'] # Materials to simulate
    densities = [1.04, 1.0, 1.92, 1.03] # g/cm^3
     
    # G4_WATER'             9.0 MeV  #8.9,8.8,8.7
    # G4_BONE_CORTICAL_ICRP 12.0 MeV  #11.9,11.8,11.7,11.6
    # G4_TISSUE_SOFT_ICRP   9.2 MeV  #9.1,9.0,8.9
    # G4_LUNG_ICRP          9.2 MeV  #9.1,9.0,8.9
        
    voxelPhaseFile = baseFolder[methodTable]['voxelPhaseFile']  # TOPAS parameter file
    fileName = baseFolder[methodTable]['fileName']  # TOPAS output file
    savePathTrans = baseFolder[methodTable]['savePathTrans']  # Save path for transformed probabilities
    savePathNorm = baseFolder[methodTable]['savePathNorm']  # Save path for normalized probabilities    
    
    modifyInputParameters(voxelPhaseFile, numberOfProtons)     
    resultsTableTrans = {}
    resultsTableNorm = {}
    
    thetaMaxArray = {mat: {} for mat in materials}
    thetaMinArray = {mat: {} for mat in materials}
    energyMinArray = {mat: {} for mat in materials}
    energyMaxArray = {mat: {} for mat in materials}
    
    # Star time of the simulation
    startTime = time.time()
        
    for material in materials:

        # Modify the material in the TOPAS file
        modifyMaterial(voxelPhaseFile, material, methodTable)
        resultsTableTrans[material] = {}
        resultsTableNorm[material] = {}
            
        for energy in energies:  
            modifyBeamEnergy(voxelPhaseFile, energy)  
            runTopas(voxelPhaseFile, dataPath)
            # Retrieve energy and angle distributions probabilities
            if material == 'G4_BONE_CORTICAL_ICRP':
                if energy < 11.6:
                    finalProbabilitiesTrans = np.zeros((numberOfBins, numberOfBins), dtype=np.float32)
                    finalProbabiltiesNorm = np.zeros((numberOfBins, numberOfBins), dtype=np.float32)
                    maxTheta = 0.0
                    minTheta = 0.0
                    finalEnergyMin = 0.0
                    finalEnergyMax = 0.0
                else:
                    finalProbabilitiesTrans, finalProbabiltiesNorm, maxTheta, minTheta, finalEnergyMin, finalEnergyMax = calculateAngleEnergyProbabilities(fileName, numberOfBins)
            elif material == 'G4_TISSUE_SOFT_ICRP' or material == 'G4_LUNG_ICRP':
                if energy < 8.9:
                    finalProbabilitiesTrans = np.zeros((numberOfBins, numberOfBins), dtype=np.float32)
                    finalProbabiltiesNorm = np.zeros((numberOfBins, numberOfBins), dtype=np.float32)
                    maxTheta = 0.0
                    minTheta = 0.0
                    finalEnergyMin = 0.0
                    finalEnergyMax = 0.0
                else:
                    finalProbabilitiesTrans, finalProbabiltiesNorm, maxTheta, minTheta, finalEnergyMin, finalEnergyMax = calculateAngleEnergyProbabilities(fileName, numberOfBins)
            else:
                finalProbabilitiesTrans, finalProbabiltiesNorm, maxTheta, minTheta, finalEnergyMin, finalEnergyMax = calculateAngleEnergyProbabilities(fileName, numberOfBins)
            
            # Store in dictionary for NPZ saving   
            resultsTableTrans[material][energy] = finalProbabilitiesTrans
            resultsTableNorm[material][energy] = finalProbabiltiesNorm
                
            # Store the maximum angle and minimum energy for each material
            thetaMaxArray[material][energy] = maxTheta
            thetaMinArray[material][energy] = minTheta
            energyMinArray[material][energy] = finalEnergyMin
            energyMaxArray[material][energy] = finalEnergyMax
                            
    # Save to CUDA-optimized NPZ
    saveToNPZ(resultsTableTrans, resultsTableNorm, materials, energies, thetaMaxArray, thetaMinArray, 
              energyMinArray, energyMaxArray, savePathTrans, savePathNorm)
    
    # End time of the simulation
    endTime = time.time()
    print(f"Simulation completed in {endTime - startTime:.2f} seconds for method {methodTable} with {numberOfProtons} protons.")