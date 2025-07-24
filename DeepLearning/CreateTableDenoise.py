import re
import pandas as pd
import subprocess
import numpy as np
from pathlib import Path
import argparse
import os
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
        

def modifySeed(filePath, newSeed):
    """
    Modify the seed in a TOPAS input file and save it with the same name.

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - newSeed (int): New seed value to replace in the file.
    """
    patternSeed = r"(i:Ts/Seed = )(\d+)"

    with open(filePath, 'r') as file:
        content = file.read()
    
    # Replace the seed in the matched line
    updatedFile = re.sub(patternSeed, rf"i:Ts/Seed = {newSeed}", content)
    
    with open(filePath, 'w') as file:
        file.write(updatedFile)

    
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

def modifyFractionsAndDensity(filePath, components, density):
    print(f'Lenght of components: {len(components)}')
    patternComponents = (
        r'^(uv:Ma/MyMixture/Fractions = 16\s+)'
        r'((?:-?\d*\.?\d+(?:[eE][+-]?\d+)?\s+){15}'
        r'-?\d*\.?\d+(?:[eE][+-]?\d+)?)'
        r'\s*$'
    )
    fractions_str = " ".join(f"{x:.6f}" for x in components)

    def replace_fractions(match):
        return match.group(1) + fractions_str
    
    patternDensity = r'^(d:Ma/MyMixture/Density = )-?\d*\.?\d+(?:[eE][+-]?\d+)?(\s*g/cm3\s*)$'
    
    def replace_density(match):
        return f"{match.group(1)}{density:.6f}{match.group(2)}"
    
    with open(filePath, 'r') as file:
        content = file.read()
    
    content = re.sub(patternComponents, replace_fractions, content, flags=re.MULTILINE)
    content = re.sub(patternDensity, replace_density, content, flags=re.MULTILINE)
    
    with open(filePath, 'w') as file:
        file.write(content)
        

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


def generateEnergies(totalPoints,
                      ranges=[(200, 125), (125, 50), (50, 12)],
                      proportions=[0.25, 0.35, 0.40],
                      method='linspace',
                      seed=None):
    """
    Generate unique energies divided into ranges with given proportions.

    Parameters:
    - totalPoints (int): Total number of energy points.
    - ranges (list of (start, end)): Energy ranges in MeV.
    - proportions (list of float): Fraction of energies per range.
    - method (str): 'linspace' or 'random'.
    - seed (int or None): Random seed for reproducibility.

    Returns:
    - np.ndarray: Sorted unique energies from high to low.
    """
    assert len(ranges) == len(proportions), "Ranges and proportions must match"
    assert np.isclose(sum(proportions), 1.0), "Proportions must sum to 1"

    if seed is not None:
        np.random.seed(seed)

    energies = []
    counts = [int(round(totalPoints * p)) for p in proportions]
    
    # Adjust total if rounding mismatch
    diff = totalPoints - sum(counts)
    counts[-1] += diff

    for i, ((start, end), count) in enumerate(zip(ranges, counts)):
        if count <= 0:
            continue

        if method == 'linspace':
            # Exclude endpoint unless last segment
            includeEndpoint = (i == len(ranges) - 1)
            sub = np.linspace(start, end, count, endpoint=includeEndpoint)
        elif method == 'random':
            sub = np.random.uniform(min(start, end), max(start, end), count)
        else:
            raise ValueError("Method must be 'linspace' or 'random'")
        energies.append(sub)

    energies = np.concatenate(energies)
    energies = np.unique(energies)  # Ensure strict uniqueness
    energies = np.sort(energies)[::-1]  # Descending order
    return energies


def selectRandomMaterialIndices(totalMaterials, numToSelect, seed=None):
    """
    Selects a list of unique random indices from 0 to totalMaterials - 1.

    Parameters:
    - totalMaterials (int): Total number of available materials.
    - numToSelect (int): Number of unique materials to select.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - np.ndarray: Array of selected indices.
    """
    if seed is not None:
        np.random.seed(seed)
        
    if totalMaterials < numToSelect:
        raise ValueError("numToSelect must be less than or equal to totalMaterials")

    return np.random.choice(totalMaterials, numToSelect, replace=False)


def selectMaterialIndices(totalMaterials, numToSelect):
    """
    Selects `numToSelect` indices from the input `totalMaterials` array,
    such that the corresponding material values are spaced as uniformly as
    possible across the range of unique material values.

    Parameters:
    - totalMaterials (np.ndarray): Array of material values (e.g., CT numbers),
                                   possibly with duplicates.
    - numToSelect (int): Number of indices to select.

    Returns:
    - np.ndarray: Array of indices into `totalMaterials`, corresponding to
                  uniformly spaced unique material values.
    """
    uniqueVals, uniqueIndices = np.unique(totalMaterials, return_index=True)
    sortedIndices = uniqueIndices[np.argsort(uniqueVals)]

    if len(sortedIndices) < numToSelect:
        raise ValueError("numToSelect must be less than or equal to the number of unique materials available.")

    # Uniformly space the selection over sorted unique material values
    selectedPositions = np.linspace(0, len(sortedIndices) - 1, numToSelect, dtype=int)
    selectedIndices = sortedIndices[selectedPositions]

    return selectedIndices


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


if __name__ == "__main__":  
    # Variables    
    methodTable = 'sheet'  # 'sheet' or 'sphere'
    dataPath = '~/G4Data/'
    materialsPath = './Materials.txt'
    saveFolder = './DenoiseTable/'
    voxelPhaseFile = './SheetVoxelByComponent.txt'

    baseFolder = {
        'sheet': {
            'voxelPhaseFile': './SheetVoxelByComponent.txt',
            'fileName': './OutputVoxelSheet.phsp',
            'saveFileNameTrans': 'DenoisingDataTransSheet.npz',
            'saveFileNameNorm': 'DenoisingDataNormSheet.npz'
        },
        'sphere': {
            'voxelPhaseFile':  './SphereVoxelByComponent.txt',
            'fileName': './OutputVoxelSphere.phsp',
            'saveFileNameTrans': 'DenoisingDataTransSphere.npz',
            'saveFileNameNorm': 'DenoisingDataNormSphere.npz'
        }
    }
    
    # Define paths based on the method table
    voxelPhaseFile = baseFolder[methodTable]['voxelPhaseFile']  # TOPAS parameter file
    fileName = baseFolder[methodTable]['fileName']  # TOPAS output file
    
    # Construct full save paths
    savePathTrans = os.path.join(saveFolder, baseFolder[methodTable]['saveFileNameTrans'])
    savePathNorm = os.path.join(saveFolder, baseFolder[methodTable]['saveFileNameNorm'])

    Path(saveFolder).mkdir(parents=True, exist_ok=True)

    # Input parameters
    numberOfBins = 100
    
    # Load materials file 
    properties = np.load("InterpolatedCompositions.npy")  # shape: (N, 18)

    # Split into 3 arrays
    elementFractions = properties[:, :16] 
    densities = properties[:, 16]            
    huValues = properties[:, 17]    
    
    # Energies from 200 to 15 MeV in steps of 1
    # energies = np.arange(200, 14, -1)  # MeV
    energies = np.linspace(200, 15, 50) 
    numberOfMaterials = elementFractions.shape[0]
    numberOfEnergies = energies.shape[0]
           
    # Re run each simulation three times with differnt seeds and number of protons
    numberOfRuns = 2 # First run variable x_1 (noisy), x_2 (noisy), x_clean (clean)
    numberOfProtons = [10000, 10000, 
                       # 10000000
                    ]  # Number of protons for each run
    
    # Initialize arrays to store final probabilities
    histogramsNorm = np.empty((numberOfRuns, numberOfMaterials, numberOfEnergies, numberOfBins, numberOfBins))
    histogramsTrans = np.empty_like(histogramsNorm)
    
    thetaMaxArray = np.empty((numberOfRuns, numberOfMaterials, numberOfEnergies))
    thetaMinArray = np.empty_like(thetaMaxArray)
    finalEnergyMinArray = np.empty_like(thetaMaxArray)
    finalEnergyMaxArray = np.empty_like(thetaMaxArray)
    
    # Start TOPAS simulation
    print(f'Starting simulation')
    startTime = time.time()
    
    # Loop through each selected material index
    for i_mat in range(numberOfMaterials):
        # Extract the components and density for the specified index
        components = elementFractions[i_mat]
        density = densities[i_mat]
        modifyFractionsAndDensity(voxelPhaseFile, components, density)

        for i_E, energy in enumerate(energies):
            modifyBeamEnergy(voxelPhaseFile, energy)
            
            for run in range(numberOfRuns):
                # Modify the seed and number of protons for each run
                seed = np.random.randint(0, 2**31 - 1)
                modifySeed(voxelPhaseFile, seed)
                modifyInputParameters(voxelPhaseFile, numberOfProtons[run])
                runTopas(voxelPhaseFile, dataPath)
                
                # Return values for each run
                finalProbabilitiesTrans, finalProbabilitiesNorm, maxTheta, minTheta, finalEnergyMin, finalEnergyMax = calculateAngleEnergyProbabilities(fileName, numberOfBins)
                
                # Store the results in the histograms
                histogramsTrans[run, i_mat, i_E] = finalProbabilitiesTrans
                histogramsNorm[run, i_mat, i_E] = finalProbabilitiesNorm
                thetaMaxArray[run, i_mat, i_E] = maxTheta
                thetaMinArray[run, i_mat, i_E] = minTheta
                finalEnergyMinArray[run, i_mat, i_E] = finalEnergyMin
                finalEnergyMaxArray[run, i_mat, i_E] = finalEnergyMax
    
    # Save the results to .npz files
    np.savez(savePathTrans, histograms=histogramsTrans, HU = huValues,
             energies = energies, rho = densities)
    
    np.savez(savePathNorm, histograms=histogramsNorm, HU = huValues,
             energies = energies, rho = densities, 
             thetaMax = thetaMaxArray, thetaMin = thetaMinArray, 
             energyMin = finalEnergyMinArray, energyMax = finalEnergyMaxArray)     
       
    # End time of the simulation
    endTime = time.time()
    print(f"Simulation time: {endTime - startTime:.12f} seconds")
