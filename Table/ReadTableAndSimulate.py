import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import os
import time
import argparse
from collections import namedtuple  # Supposedly more faster
# from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor, as_completed

params = {
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,      
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'legend.fontsize': 14
}
pylab.rcParams.update(params)  # Apply changes

def sampleReverseCalculate(data, material, energy, angleRange, energyRange, materialToIndex):
    """
    Sample (angle, energy) values from a 2D histogram and reverse the variable change.
    
    Parameters:
    - data (dict): Dictionary containing the histogram data.
    - material (str): Name of the material the particle is in.
    - energy (float): Initial energy of the proton.
    - angleRange (tuple): Range of angles to sample from.
    - energyRange (tuple): Range of energies to sample from.
    - materialToIndex (dict): Dictionary mapping material names to indices.
    
    Returns:
    - realAngle (float): Real angle value.
    - realEnergy (float): Real energy value.
    """

    # Extract metadata
    probTable = data['prob_table']
    energies = data['energies']
    
    # Round energy, no interpolation
    energy = np.round(energy, 1)

    # Validate and locate material and energy
    if material not in materialToIndex:
        raise ValueError(f"Material '{material}' not found in the dataset.")
    if energy not in energies:
        raise ValueError(f"Energy '{energy}' not found in the dataset.")

    materialIdx = materialToIndex[material]
    energyIdx = np.where(energies == energy)[0][0]

    # Retrieve the histogram for the specified material and energy
    hist = probTable[materialIdx, energyIdx]
    
    angleSample, energySample = sampleFromHist(hist, angleRange, energyRange)
    realAngle, realEnergy = reverseVariableChange(energy, angleSample, energySample)
    
    return realAngle, realEnergy

def sampleReverseCalculateInterpolation(data, material, energy, angleRange, energyRange, materialToIndex):
    """
    Sample (angle, energy) values from a 2D histogram and apply interpolation and reverse the variable change.
    
    Parameters:
    - data (dict): Dictionary containing the histogram data.
    - material (str): Name of the material the particle is in.
    - energy (float): Initial energy of the proton.
    - angleRange (tuple): Range of angles to sample from.
    - energyRange (tuple): Range of energies to sample from.
    - materialToIndex (dict): Dictionary mapping material names to indices.
    
    Returns:
    - realAngle (float): Real angle value.
    - realEnergy (float): Real energy value.
    """
    
    # Extract metadata
    probTable = data.prob_table
    energies = np.sort(data.energies)

    # Get material index
    if material not in materialToIndex:
        raise ValueError(f"Material '{material}' not found in data.")
    materialIdx = materialToIndex[material]
    
    # Find surrounding energy indices
    if energy < energies[0] or energy > energies[-1]:
        raise ValueError(f"Energy {energy} out of bounds ({energies[0]} - {energies[-1]})")

    lowerIndex = np.searchsorted(energies, energy) -1
    upperIndex = lowerIndex + 1
    
    # Clamp to valid range
    lowerIndex = max(0, lowerIndex)
    upperIndex = min(len(energies) - 1, upperIndex)
    
    energyLow = energies[lowerIndex]
    energyUp = energies[upperIndex]
    
    probLow = probTable[materialIdx, lowerIndex]
    probHigh = probTable[materialIdx, upperIndex]
    
    # It computes a weighted average of the two histograms based on how close energy is to each:
    if energyUp == energyLow:
        interpolatedHist = probLow
    else:
        weight = (energy - energyLow) / (energyUp - energyLow)
        interpolatedHist = (1 - weight) * probLow + weight * probHigh
        
    angleSample, energySample = sampleFromHist(interpolatedHist, angleRange, energyRange)
    realAngle, realEnergy = reverseVariableChange(energy, angleSample, energySample)
    
    return realAngle, realEnergy

def sampleFromHist(hist, angleRange, energyRange):
    """
    Sample (angle, energy) values from a 2D histogram.

    Parameters:
    - hist (2D np.array): Normalized joint probability histogram.
    - angleRange (tuple): Min/max range of transformed angles.
    - energyRange (tuple): Min/max range of transformed energy losses.

    Returns:
    - angle (float): Sampled angle (transformed).
    - energy (float): Sampled energy (transformed).
    """
    # Flatten and sample using probabilities
    flatHist = hist.flatten()
    
    # Sampling one histogram bin at a time consider pre-generating samples or using faster sampling methods like np.searchsorted() with cumulative probabilities
    # idx = np.random.choice(len(flatHist), p=flatHist)
    cumsum = np.cumsum(flatHist)
    rand_val = np.random.rand()
    idx = np.searchsorted(cumsum, rand_val, side='right')

    # Convert back to 2D index
    angleBins, energyBins = hist.shape
    angleIdx, energyIdx = np.unravel_index(idx, (angleBins, energyBins))

    # Convert indices to actual values (bin centers)
    angleStep = (angleRange[1] - angleRange[0]) / angleBins
    energyStep = (energyRange[1] - energyRange[0]) / energyBins

    angle = angleRange[0] + (angleIdx + 0.5) * angleStep
    energy = energyRange[0] + (energyIdx + 0.5) * energyStep

    return angle, energy

def reverseVariableChange(initialEnergy, angle, energy):
    """
    Reverse the variable change applied to the energy loss value and angle.

    Parameters:
    - energy (float): initial energy.

    Returns:
    - realEnergy (float): energy after reversing the variable change.
    - realAngle (float): angle after reversing the variable change.
    """
    # Reverse the angle change
    realAngle = angle / np.sqrt(initialEnergy)  
    # Reverse the energy change
    realEnergy = initialEnergy * (1 - np.exp(energy * np.sqrt(initialEnergy)))  
    
    return realAngle, realEnergy

def variableChange(energy, angle, energyloss):
    """
    Apply the variable change to the energy loss value and angle.
    
    Parameters:
    - energy (float): initial energy.
    - angle (float): angle value.
    - energyloss (float): energy loss value.
    
    Returns:
    - energyChange (float): energy loss value after applying the variable change.
    - angleChange (float): angle value after applying the variable change.
    """
    # Apply the variable change
    energyChange = np.log((energy - energyloss) / energy) / np.sqrt(energy)
    angleChange = angle * np.sqrt(energy) 
    
    return energyChange, angleChange

def plotSampledDistribution(hist, numSamples, angleRange=(0, 70), energyRange=(-0.57, 0)):
    # Generate samples
    angles = []
    energies = []
    
    for h in range(numSamples):
        print(f"Sampling {h+1} ")
        angle, energy = sampleFromHist(hist, angleRange, energyRange)
        angles.append(angle)
        energies.append(energy)

    # Alternatively, we can also plot a 2D histogram of the samples
    plt.figure(figsize=(8, 6))
    plt.hist2d(angles, energies, bins=70, range=[angleRange, energyRange], cmap='Blues')
    plt.colorbar(label='Frequency')
    plt.xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
    plt.ylabel(r'$\frac{ln((E_i-E_f)/E_i)}{\sqrt{E_i}}$ (MeV$^{-1/2}$)')
    plt.savefig('SampledDistribution.pdf')
    plt.close()
    

def checkIfInsideBigVoxel(position, voxelSize):
    x, y, z = position
    return (
        -voxelSize <= x <= voxelSize and
        -voxelSize <= y <= voxelSize and
        -voxelSize <= z <= voxelSize
    )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--interp", action="store_true", help="Enable interpolation")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    nProtonsTable = 100000
    npzPath = f'./Table/4DTable{nProtonsTable}.npz' 
    samplingN = 100
    
    material = 'G4_WATER'
    initialEnergy = 200.0
    sameMaterial = True  # Have we changed the material?
    numberOfBins = 100
    savePath = './MySimulation/'
    
    # Define the ranges used in the histogram transformation
    angleRange = (0, 70)  # degrees * sqrt(E_i)
    energyRange = (-0.57, 0)  # log10((E_i - E_f)/E_i) / sqrt(E_i)

    bigVoxelSize = 1e3 / 2 # (mm) maximum size for the simulation voxel
    
    # Load the whole table
    data = np.load(npzPath, allow_pickle=True)
    materialToIdx = {mat: idx for idx, mat in enumerate(data["materials"])}

    # Plot the sampled distribution
    # plotSampledDistribution(hist, samplingN)
    
    # Calculate time of simulation
    starTime = time.time()
    
    for i in range(samplingN):
        angleChange, energyChange = 0, 0
        energy = initialEnergy
        position = np.array([0, 0, -bigVoxelSize], dtype=float)
        velocity = np.array([0, 0, 1], dtype=float)
        dt = 1 / 3 # step size in voxel units |v| = 1
        
        xAxis = np.array([1, 0, 0], dtype=float)
        yAxis = np.array([0, 1, 0], dtype=float)
        zAxis = np.array([0, 0, 1], dtype=float)

        while energy > 0 and checkIfInsideBigVoxel(position, bigVoxelSize):
            if interp:
                realAngle, realEnergy = sampleReverseCalculateInterpolation(data, material, energy, angleRange, energyRange, materialToIndex)
            else:
                realAngle, realEnergy = sampleReverseCalculate(data, material, energy, angleRange, energyRange, materialToIndex)

            energyChange, angleChange = variableChange(energy, realAngle, realEnergy)
            if debug:
                print(f"[Particle {particleId}] Energy: {realEnergy}, Angle: {realAngle}")
            energy = realEnergy

            phi = np.random.uniform(0, 2 * np.pi)
            newDirection = np.array([
                np.sin(np.radians(realAngle)) * np.cos(phi),
                np.sin(np.radians(realAngle)) * np.sin(phi),
                np.cos(np.radians(realAngle))
            ])
            
            rotationMatrix = np.stack([xAxis, yAxis, zAxis], axis=1)
            direction = rotationMatrix @ newDirection
            direction = direction / np.linalg.norm(direction)

            position += direction * dt

            if sameMaterial:
                continue
