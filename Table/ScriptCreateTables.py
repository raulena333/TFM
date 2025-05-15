import re
import os
import subprocess
import numpy as np
from scipy.interpolate import interp1d

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
        

def calculateAngleEnergyProbabilities(fileName, material, numberOfBins):
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

    maskTrans = (logE < threshold[0]) & (logE > threshold[1])
    dirX_T = finalDirX[maskTrans]
    dirY_T = finalDirY[maskTrans]
    sign_T = isSign[maskTrans]
    logE_T = logE[maskTrans]
    finalEnergy = finalEnergy[maskTrans]

    dirZ_T = np.sqrt(np.clip(1 - dirX_T**2 - dirY_T**2, 0, 1))
    dirZ_T[sign_T == 0] *= -1
    angle = np.degrees(np.arccos(np.clip(dirZ_T, -1.0, 1.0))) 
    angle_T = angle * np.sqrt(energy)

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

    
if __name__ == "__main__":  
    # Variables    
    numberOfProtons = 10000 # Number of protons to simulate
    dataPath = '~/G4Data/'
    voxelPhaseFile = './MyVoxelPhaseSpace.txt'
    fileName = 'OutputVoxel.phsp'
    
    # Input parameters
    numberOfBins = 100
    initialEnergy = 200.
    finalEnergy = 9.
    stepEnergy = 0.2
    energies = np.round(np.arange(initialEnergy, finalEnergy, -stepEnergy), 1)
      
    materials = [#'G4_LUNG_ICRP', 
                'G4_WATER', 
                #'G4_BONE_CORTICAL_ICRP', 'G4_TISSUE_SOFT_ICRP' 
                ] # Materials to simulate
    densities = [1.04, 1.0, 1.92, 1.03] # g/cm^3

    savePathTrans = f'./Table/4DTableTrans.npz'
    savePathNorm = f'./Table/4DTableNorm.npz'
        
    modifyInputParameters(voxelPhaseFile, numberOfProtons)     
    resultsTableTrans = {}
    resultsTableNorm = {}
    
    thetaMaxArray = {mat: {} for mat in materials}
    thetaMinArray = {mat: {} for mat in materials}
    energyMinArray = {mat: {} for mat in materials}
    energyMaxArray = {mat: {} for mat in materials}
        
    for material in materials:

        # Modify the material in the TOPAS file
        modifyMaterial(voxelPhaseFile, material)
        resultsTableTrans[material] = {}
        resultsTableNorm[material] = {}
            
        for energy in energies:  
            modifyBeamEnergy(voxelPhaseFile, energy)  
            runTopas(voxelPhaseFile, dataPath)
            # Retrieve energy and angle distributions probabilities
            if material == 'G4_BONE_CORTICAL_ICRP':
                if energy < 12:
                    finalProbabilitiesTrans = np.zeros((numberOfBins, numberOfBins), dtype=np.float32)
                    finalProbabiltiesNorm = np.zeros((numberOfBins, numberOfBins), dtype=np.float32)
                    maxTheta = 0.0
                    minTheta = 0.0
                    finalEnergyMin = 0.0
                    finalEnergyMax = 0.0
                else:
                    finalProbabilitiesTrans, finalProbabiltiesNorm, maxTheta, minTheta, finalEnergyMin, finalEnergyMax = calculateAngleEnergyProbabilities(fileName, material, numberOfBins)
            elif material == 'G4_TISSUE_SOFT_ICRP' or material == 'G4_LUNG_ICRP':
                if energy < 9.5:
                    finalProbabilitiesTrans = np.zeros((numberOfBins, numberOfBins), dtype=np.float32)
                    finalProbabiltiesNorm = np.zeros((numberOfBins, numberOfBins), dtype=np.float32)
                    maxTheta = 0.0
                    minTheta = 0.0
                    finalEnergyMin = 0.0
                    finalEnergyMax = 0.0
                else:
                    finalProbabilitiesTrans, finalProbabiltiesNorm, maxTheta, minTheta, finalEnergyMin, finalEnergyMax = calculateAngleEnergyProbabilities(fileName, material, numberOfBins)
            else:
                finalProbabilitiesTrans, finalProbabiltiesNorm, maxTheta, minTheta, finalEnergyMin, finalEnergyMax = calculateAngleEnergyProbabilities(fileName, material, numberOfBins)
            
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