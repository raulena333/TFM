import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing
import logging

# python3 -m cProfile -o profile_results.prof ReadTableAndSimulateParallel.py
# snakeviz profile_results.prof
# gprof2dot -f pstats profile_results.prof | dot -Tpng -o profile.png

# Plot styling
params = {
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'legend.fontsize': 14
}
pylab.rcParams.update(params)

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

def checkIfInsideBigVoxel(position, voxelSize):
    x, y, z = position
    return (
        -voxelSize <= x <= voxelSize and
        -voxelSize <= y <= voxelSize and
        -voxelSize <= z <= voxelSize
    )

import numpy as np

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
    
    # Round energy to 1 decimal place
    energy = np.round(energy, 1)
    
    if energy < 15.:     
        return 0, 0
    else:
        # Validate and locate material
        if material not in materialToIndex:
            raise ValueError(f"Material '{material}' not found in the dataset.")
        
        # Find the closest energy in the dataset (rounded to 1 decimal place)
        closest_energy_idx = np.argmin(np.abs(np.round(energies, 1) - energy))  # Find the index of the closest match
        closest_energy = energies[closest_energy_idx]

        # Retrieve the histogram for the specified material and closest energy
        materialIdx = materialToIndex[material]
        energyIdx = closest_energy_idx
        
        # Retrieve the histogram for the specified material and energy
        hist = probTable[materialIdx, energyIdx]
        
        # Perform histogram sampling and reverse variable change
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
    
    if energy < 15.:     
        return 0, 0
    else:
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

def simulateOneParticle(args):
    (
        particleId, data, material, initialEnergy, angleRange, energyRange,
        materialToIndex, interp, debug, bigVoxelSize
    ) = args

    try:
        angleChange, energyChange = 0, 0
        energy = initialEnergy
        initialPosition = np.array([0, 0, -bigVoxelSize], dtype=float)
        velocity = np.array([0, 0, 1], dtype=float)
        dt = 1 / 3  # step size in voxel units |v| = 1
        sameMaterial = True
        
        if debug:
            print(f"[Particle {particleId}], Initial Position: {initialPosition}")
        
        position = initialPosition + velocity * dt

        while energy > 0 and checkIfInsideBigVoxel(position, bigVoxelSize):
            if interp:
                realAngle, realEnergy = sampleReverseCalculateInterpolation(
                    data, material, energy, angleRange, energyRange, materialToIndex
                )
            else:
                realAngle, realEnergy = sampleReverseCalculate(
                    data, material, energy, angleRange, energyRange, materialToIndex
                )

            energyChange, angleChange = variableChange(initialEnergy, realAngle, realEnergy)

            if debug:
                print(f"Position: {position}, Energy: {realEnergy}, Angle: {realAngle}")

            energy = realEnergy

            phi = np.random.uniform(0, 2 * np.pi)
            theta = np.radians(realAngle)

            # Get perpendicular vector
            perp = np.cross(velocity, [0, 0, 1])
            norm = np.linalg.norm(perp)
            if norm < 1e-8:
                perp = np.array([1, 0, 0])
            else:
                perp /= norm

            # Use Rodrigues' formula to rotate v by theta in random direction phi
            # Generate vector in cone around v
            w = (
                np.cos(theta) * velocity +
                np.sin(theta) * (np.cos(phi) * np.cross(perp, velocity) + np.sin(phi) * perp)
            )
            
            velocity = w / np.linalg.norm(w)
            position += velocity * dt

            if sameMaterial:
                continue

        if debug:
            print(f"[Particle {particleId}] Final Angle: {angleChange}, Energy Loss: {energyChange}")
            print("--------------------------------------\n")
            

        return angleChange, energyChange

    except ValueError as ve:
        logging.error(f"simulateOneParticle failed: {ve} | particleId: {particleId}, material: {material}, energy: {energy}")
        return None

    except Exception as e:
        logging.exception(f"Unexpected error for particleId {particleId} | material: {material}, energy: {energy} | {e}")
        return None



# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interp", action="store_true", help="Enable interpolation")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    logging.basicConfig(filename="simulation_errors.log", level=logging.ERROR)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    nProtonsTable = 100000
    npzPath = f'./Table/4DTable{nProtonsTable}.npz'
    samplingN = [10, 100, 500, 1000, #5000, 10000, 50000, 100000
                 ]
    numberOfBins = 20
    material = 'G4_WATER'
    initialEnergy = 200.0
    savePath = './Plots/'
    angleRange = (0, 70)
    energyRange = (-0.57, 0)
    bigVoxelSize = 33

    rawData = np.load(npzPath, allow_pickle=True)

    # Fully extract arrays here
    prob_table = rawData['prob_table']
    materials = rawData['materials']
    energies = rawData['energies']
    
    # Store in plain dict — safe for multiprocessing
    data = {
        'prob_table': np.array(prob_table),
        'materials': list(materials),
        'energies': np.array(energies)
    }
    rawData.close()

    materialToIndex = {mat: idx for idx, mat in enumerate(data['materials'])}
    angleDistribution = []
    energyDistribution = []
    
    timeOfHistories = []

    for sampling in samplingN:
        
        starTime = time.time()
        # Create args list to feed into pool.map
        argsList = [
            (
                l, data, material, initialEnergy, angleRange, energyRange,
                materialToIndex, args.interp, args.debug, bigVoxelSize
            )
            for l in range(sampling)
        ]

        numThreads = multiprocessing.cpu_count()
        with Pool(processes=numThreads) as pool:
            results = list(tqdm(pool.imap(simulateOneParticle, argsList), total=sampling, desc="Simulating"))

        for angle, energy in results:
            angleDistribution.append(angle)
            energyDistribution.append(energy)
        endTime = time.time()
        timeOfHistories.append(endTime - starTime)
        print(f"Simulation time: {endTime - starTime:.2f} seconds")
        print() 
        
    # Plot
    plt.plot(samplingN, timeOfHistories, linestyle='-',marker = '.', color="black")      
    plt.xlabel("Number of Histories")
    plt.ylabel("Time (s)")
    plt.tight_layout()  
    plt.savefig(f"{savePath}HistoryTimePlot.pdf")
    # plt.show()
    plt.close()

    # Plot histograms
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    sns.histplot(energyDistribution, bins=numberOfBins, edgecolor="black", color='orange', kde=False, ax=axs[0]) 
    axs[0].set_xlabel(r'$\frac{ln((E_i-E_f)/E_i)}{\sqrt{E_i}}$ (MeV$^{-1/2}$)')
    axs[0].set_title('Final Energy distribution')
    axs[0].set_yscale('log')
            
    sns.histplot(angleDistribution, bins=numberOfBins, edgecolor="black", color='red', kde=False, ax=axs[1])
    axs[1].set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
    axs[1].set_title('Final Angles distribution')
    axs[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{savePath}MySimulationHistograms.pdf')
    plt.close(fig)

    
    hist1, xedges1, yedges1 = np.histogram2d(angleDistribution, energyDistribution, bins=numberOfBins)
    finalProbabilities = hist1 / np.sum(hist1)

    fig2, axs2 = plt.subplots(figsize=(8, 6))
    h1 = axs2.pcolormesh(xedges1, yedges1, finalProbabilities.T, cmap='Reds', shading='auto')
    fig2.colorbar(h1, ax=axs2, label='Probability')
    axs2.set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
    axs2.set_ylabel(r'$ln((E_i-E_f)/E_i)\sqrt{E_i}$ (MeV$^{-1/2}$)')

    plt.tight_layout()
    plt.savefig(f'{savePath}2DMySimulationHistograms.pdf')
    plt.close(fig2) 