import re
import pandas as pd
import subprocess
import numpy as np
from pathlib import Path
import argparse
import os
import time
from tqdm import tqdm

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

def calculateAngleEnergyProbabilities(
    fileName: str,
    numberOfBins: int,
    globalMinTheta: float = None,
    globalMaxTheta: float = None,
    globalMinEnergy: float = None,
    globalMaxEnergy: float = None
):
    """
    Calculates and returns 2D histograms with consistent normalization.
    
    This function handles both the first run (by calculating min/max) and 
    subsequent runs (by using the provided global min/max values).

    Parameters:
    - fileName (str): Path to the TOPAS output file.
    - numberOfBins (int): Number of bins for 2D histogram in both angle and energy.
    - globalMin/Max... (Optional[float]): Optional normalization values from a previous run.
    
    Returns:
    - finalProbabilitiesTrans (ndarray): 2D histogram (angle vs transformed log energy).
    - finalProbabilitiesNorm (ndarray): 2D histogram (normalized angle vs. normalized energy).
    - finalMaxTheta (float): Maximum angle used for normalization.
    - finalMinTheta (float): Minimum angle used for normalization.
    - finalEnergyMin (float): Minimum final energy for normalization.
    - finalEnergyMax (float): Maximum final energy for normalization.
    """
    # Thresholds for transformed energy and angular filtering
    threshold = [0, -0.6, 70]
    
    # Check if the file is empty or doesn't exist.
    if not os.path.exists(fileName) or os.path.getsize(fileName) == 0:
        print(f"Warning: File {fileName} is empty or not found. Returning zero arrays.")
        zeros_2d = np.zeros((numberOfBins, numberOfBins))
        return zeros_2d, zeros_2d, 0.0, 0.0, 0.0, 0.0

    # Load data
    try:
        data = np.loadtxt(fileName)
    except ValueError:
        print(f"Warning: Could not load data from {fileName} due to malformed content. Returning zero arrays.")
        zeros_2d = np.zeros((numberOfBins, numberOfBins))
        return zeros_2d, zeros_2d, 0.0, 0.0, 0.0, 0.0

    print(f'{fileName} loaded successfully.')
    
    # Check if the loaded data is an empty array or a single event (1D array).
    if data.size == 0 or data.ndim == 1:
        print(f"Warning: Data in {fileName} is empty or contains only a single event. Returning zero arrays.")
        zeros_2d = np.zeros((numberOfBins, numberOfBins))
        return zeros_2d, zeros_2d, 0.0, 0.0, 0.0, 0.0

    finalDirX, finalDirY, finalEnergy, isSign, initialEnergy = data[:, [3, 4, 5, 8, 10]].T
    energy = np.mean(initialEnergy)
 
    # --- DETERMINE NORMALIZATION VALUES ---
    # This logic handles both the first and subsequent runs.
    
    dirZ = np.sqrt(np.clip(1 - finalDirX**2 - finalDirY**2, 0, 1))
    dirZ[isSign == 0] *= -1
    angle = np.degrees(np.arccos(np.clip(dirZ, -1.0, 1.0)))
    
    # If this is the first run, calculate the min/max from this data.
    if globalMinTheta is None:
        globalMinTheta = np.min(angle)
        globalMaxTheta = np.max(angle)
        globalMinEnergy = np.min(finalEnergy)
        globalMaxEnergy = np.max(finalEnergy)

    # --- TRANSFORMED ENERGY PATH ---
    logE = np.log((initialEnergy - finalEnergy) / initialEnergy)
    logE *= 1 / np.sqrt(initialEnergy)
    logE = np.clip(logE, threshold[1], threshold[0])

    maskTrans = (logE < threshold[0]) & (logE > threshold[1])
    logE_T = logE[maskTrans]
    angle_T = angle[maskTrans] * np.sqrt(energy)
    angle_T = np.clip(angle_T, 0, threshold[2])
    angle_mask = angle_T <= threshold[2]
    angle_T = angle_T[angle_mask]
    logE_T = logE_T[angle_mask]
    
    histTrans, _, _ = np.histogram2d(angle_T, logE_T,
                                     bins=numberOfBins,
                                     range=([0, threshold[2]], [threshold[1], threshold[0]]))
    
    # --- NORMALIZED ENERGY PATH ---
    angle_N_norm = (angle - globalMinTheta) / (globalMaxTheta - globalMinTheta)
    energy_Norm = (finalEnergy - globalMinEnergy) / (globalMaxEnergy - globalMinEnergy)
    
    histNorm, _, _ = np.histogram2d(angle_N_norm, energy_Norm,
                                    bins=numberOfBins,
                                    range=([0, 1], [0, 1]))

    # --- FINAL NORMALIZATION ---
    sum_trans = np.sum(histTrans)
    if sum_trans > 0:
        finalProbabilitiesTrans = histTrans / sum_trans
    else:
        finalProbabilitiesTrans = np.zeros((numberOfBins, numberOfBins))

    sum_norm = np.sum(histNorm)
    if sum_norm > 0:
        finalProbabilitiesNorm = histNorm / sum_norm
    else:
        finalProbabilitiesNorm = np.zeros_like(histNorm)
    
    # Return the histograms and the normalization values used for record-keeping
    return finalProbabilitiesTrans, finalProbabilitiesNorm, globalMaxTheta, globalMinTheta, globalMaxEnergy, globalMinEnergy


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
    
    # Energies from 200 to 15 MeV
    # First segment: 200.4 to 25 MeV with 0.2 MeV steps
    highSegment = np.round(np.arange(200.4, 25, -0.2), 1)
    # Second segment: 25 to 8.5 MeV with 0.1 MeV steps
    lowSegment = np.round(np.arange(25, 5.6, -0.1), 1)
    energies = np.concatenate((highSegment, lowSegment))

    numberOfMaterials = elementFractions.shape[0]
    numberOfEnergies = energies.shape[0]
           
    # Re run each simulation two times with different seeds and number of protons
    numberOfRuns = 2 # First run variable x_1 (noisy), x_2 (noisy),
    numberOfProtons = [100, 100, 
                       # 10000000
                    ]  # Number of protons for each run
    
    # Initialize arrays to store final probabilities
    histogramsNorm = np.empty((numberOfRuns, numberOfMaterials, numberOfEnergies, numberOfBins, numberOfBins))
    histogramsTrans = np.empty_like(histogramsNorm)
    
    print(histogramsNorm.shape)
    print(histogramsTrans.shape)
    
    thetaMaxArray = np.empty((1, numberOfMaterials, numberOfEnergies))
    thetaMinArray = np.empty_like(thetaMaxArray)
    finalEnergyMinArray = np.empty_like(thetaMaxArray)
    finalEnergyMaxArray = np.empty_like(thetaMaxArray)
    
    # Start TOPAS simulation
    print(f'Starting simulation')
    startTime = time.time()
    
    total_iterations = numberOfMaterials * numberOfEnergies * numberOfRuns
    overall_pbar = tqdm(total=total_iterations,
                    desc='Total progress',
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    # Loop through each selected material index
    for i_mat in range(numberOfMaterials):
        components = elementFractions[i_mat]
        density = densities[i_mat]
        modifyFractionsAndDensity(voxelPhaseFile, components, density)

        for i_E, energy in enumerate(energies):
            modifyBeamEnergy(voxelPhaseFile, energy)
            
            # --- Variables to store the normalization standard for this energy/material combo ---
            # Initialize them to None for each new energy/material
            minTheta_norm = None
            maxTheta_norm = None
            minEnergy_norm = None
            maxEnergy_norm = None
            # -----------------------------------------------------------------------------------
            
            for run in range(numberOfRuns):
                seed = np.random.randint(0, 2**31 - 1)
                modifySeed(voxelPhaseFile, seed)
                modifyInputParameters(voxelPhaseFile, numberOfProtons[run])
                runTopas(voxelPhaseFile, dataPath)
                
                # --- THE CORRECT LOGIC FOR NORMALIZATION ---
                if run == 0:
                    # FIRST RUN: Calculate the normalization standard and store it
                    (finalProbabilitiesTrans, 
                    finalProbabilitiesNorm, 
                    maxTheta, minTheta, 
                    finalEnergyMax, finalEnergyMin) = calculateAngleEnergyProbabilities(
                        fileName=fileName,
                        numberOfBins=numberOfBins
                    )
                    
                    # Store these values to be used for all other runs
                    minTheta_norm = minTheta
                    maxTheta_norm = maxTheta
                    minEnergy_norm = finalEnergyMin
                    maxEnergy_norm = finalEnergyMax  
                else:
                    # SUBSEQUENT RUNS: Pass the standard from the first run
                    (finalProbabilitiesTrans, 
                    finalProbabilitiesNorm, 
                    maxTheta, minTheta, 
                    finalEnergyMax, finalEnergyMin) = calculateAngleEnergyProbabilities(
                        fileName=fileName,
                        numberOfBins=numberOfBins,
                        globalMinTheta=minTheta_norm,
                        globalMaxTheta=maxTheta_norm,
                        globalMinEnergy=minEnergy_norm,
                        globalMaxEnergy=maxEnergy_norm
                    )
                # -------------------------------------------
                
                # Store the results in the appropriate arrays
                histogramsTrans[run, i_mat, i_E] = finalProbabilitiesTrans
                histogramsNorm[run, i_mat, i_E] = finalProbabilitiesNorm
                if run == 0:
                    thetaMaxArray[run, i_mat, i_E] = maxTheta
                    thetaMinArray[run, i_mat, i_E] = minTheta
                    finalEnergyMinArray[run, i_mat, i_E] = finalEnergyMin
                    finalEnergyMaxArray[run, i_mat, i_E] = finalEnergyMax
                
                overall_pbar.update(1)
    
    overall_pbar.close()
    
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

    # Check if there is any NAN or inf in both histograms, returns False if not
    print(np.isnan(histogramsNorm).any())
    print(np.isinf(histogramsNorm).any())
    print(np.isnan(histogramsTrans).any())
    print(np.isinf(histogramsTrans).any())
    