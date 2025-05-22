import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool, cpu_count
import logging
import csv

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

def sampleFromHist(hist, angleRange, energyRange, batchSize=1):
    """
    Sample (angle, energy) values from a 2D histogram depends on the batch size. If batch size is 1, 
    returns a single value. Else, returns a tuple of arrays.

    Parameters:
    - hist (2D np.array): Normalized joint probability histogram.
    - angleRange (tuple): Min/max range of transformed angles.
    - energyRange (tuple): Min/max range of transformed energy losses.
    - batchSize (int): Number of samples to generate at once.

    Returns:
    - angle (float, array): Sampled angle (transformed).
    - energy (float, array): Sampled energy (transformed).
    """
    # Flatten and sample using probabilities
    flatHist = hist.flatten()
    flatHist /= flatHist.sum()
    angleBins, energyBins = hist.shape
    
    # Sampling one histogram bin at a time consider pre-generating samples or using faster sampling methods like np.searchsorted() with cumulative probabilities
    # idx = np.random.choice(len(flatHist), p=flatHist)
    # cumsum = np.cumsum(flatHist)
    # rand_val = np.random.rand()
    # idx = np.searchsorted(cumsum, rand_val, side='right')
    
    # Batch sampling: one-hot vectors from multinomial
    draws = np.random.multinomial(1, flatHist, size=batchSize)
    indices = np.argmax(draws, axis=1)  # Shape: (batchSize,)
    
    # Convert 1D indices to 2D bin indices
    angleIdxs, energyIdxs = np.unravel_index(indices, (angleBins, energyBins))

    # Compute bin centers
    angleStep = (angleRange[1] - angleRange[0]) / angleBins
    energyStep = (energyRange[1] - energyRange[0]) / energyBins

    angles = angleRange[0] + (angleIdxs + 0.5) * angleStep
    energies = energyRange[0] + (energyIdxs + 0.5) * energyStep

    if batchSize == 1:
        return angles[0], energies[0]  # for compatibility
    else:
        return angles, energies

def reverseVariableChange(initialEnergy, angle, energy):
    """
    Reverse the variable change applied to the energy loss value and angle.
    Supports scalar or vector input.

    Parameters:
    - energy (float): initial energy.

    Returns:
    - realEnergy (float or np.array): energy after reversing the variable change.
    - realAngle (float or np.array): angle after reversing the variable change.
    """
    # Reverse the angle change
    sqrtE = np.sqrt(initialEnergy)
    realAngle = angle / sqrtE  
    # Reverse the energy change
    realEnergy = initialEnergy * (1 - np.exp(energy * sqrtE))  
    
    return realAngle, realEnergy

def variableChange(energy, angle, energyloss):
    """
    Apply the variable change to the energy loss value and angle.
    Supports scalar or vector input.
     
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

def sampleReverseCalculate(data, material, energy, angleRange, energyRange, materialToIndex, batchSize=1):
    """
    Sample (angle, energy) values from a 2D histogram and reverse the variable change.
    Supports scalar or vector input.
    
    Parameters:
    - data (dict): Dictionary containing the histogram data.
    - material (str): Name of the material the particle is in.
    - energy (float): Initial energy of the proton.
    - angleRange (tuple): Range of angles to sample from.
    - energyRange (tuple): Range of energies to sample from.
    - materialToIndex (dict): Dictionary mapping material names to indices.
    
    Returns:
    - realAngle (float or np.array): Real angle value.
    - realEnergy (float or np.array): Real energy value.
    """

    # Extract metadata
    probTable = data['prob_table']
    energies = data['energies']
    
    # Round energy to 1 decimal place
    energy = np.round(energy, 1)
    
    if energy < 15.:
        return 0, 0 if batchSize == 1 else (np.zeros(batchSize), np.zeros(batchSize))
    
    # Validate and locate material
    if material not in materialToIndex:
        raise ValueError(f"Material '{material}' not found in the dataset.")
        
    # Find the closest energy in the dataset (rounded to 1 decimal place)
    closestEnergyIdx = np.argmin(np.abs(np.round(energies, 1) - energy))  # Find the index of the closest match

    # Retrieve the histogram for the specified material and closest energy
    materialIdx = materialToIndex[material]
        
    # Retrieve the histogram for the specified material and energy
    hist = probTable[materialIdx, closestEnergyIdx]
        
    # Perform histogram sampling and reverse variable change
    angleSample, energySample = sampleFromHist(hist, angleRange, energyRange, batchSize)
    realAngle, realEnergy = reverseVariableChange(energy, angleSample, energySample)
        
    return realAngle, realEnergy


def sampleReverseCalculateInterpolation(
    data, material, energy, angleRange, energyRange, materialToIndex, batchSize=1
):
    """
    Sample (angle, energy) values from a 2D histogram with interpolation and reverse the variable change.
    
    Parameters:
    - data (dict): Histogram data.
    - material (str): Material name.
    - energy (float): Initial energy.
    - angleRange (tuple): Angle bin range.
    - energyRange (tuple): Energy bin range.
    - materialToIndex (dict): Map material name to index.
    - batchSize (int): Number of samples to generate.
    
    Returns:
    - realAngle (float or np.array): Sampled real angles.
    - realEnergy (float or np.array): Sampled real energies.
    """

    # Extract metadata
    probTable = data['prob_table']
    energies = np.sort(data['energies'])

    if energy < 15.:
        return (0, 0) if batchSize == 1 else (np.zeros(batchSize), np.zeros(batchSize))

    # Material index lookup
    if material not in materialToIndex:
        raise ValueError(f"Material '{material}' not found in data.")
    materialIdx = materialToIndex[material]

    # Bound energy and get surrounding indices
    if energy < energies[0] or energy > energies[-1]:
        raise ValueError(f"Energy {energy} out of bounds ({energies[0]} - {energies[-1]})")

    lowerIndex = np.searchsorted(energies, energy) - 1
    upperIndex = lowerIndex + 1
    lowerIndex = max(0, lowerIndex)
    upperIndex = min(len(energies) - 1, upperIndex)

    energyLow = energies[lowerIndex]
    energyUp = energies[upperIndex]

    probLow = probTable[materialIdx, lowerIndex]
    probHigh = probTable[materialIdx, upperIndex]

    # Interpolation
    if energyUp == energyLow:
        interpolatedHist = probLow
    else:
        weight = (energy - energyLow) / (energyUp - energyLow)
        interpolatedHist = (1 - weight) * probLow + weight * probHigh

    # Sampling
    angleSample, energySample = sampleFromHist(interpolatedHist, angleRange, energyRange, batchSize)
    realAngle, realEnergy = reverseVariableChange(energy, angleSample, energySample)

    if batchSize == 1:
        return realAngle[0], realEnergy[0]
    else:
        return realAngle, realEnergy


def simulateBatchParticles(
    batchSize, data, material, initialEnergy,
    angleRange, energyRange, materialToIndex,
    interp, bigVoxelSize
):
    try:
        # Initialize state arrays
        energy = np.full(batchSize, initialEnergy)
        position = np.tile([0, 0, -bigVoxelSize], (batchSize, 1)).astype(float)
        initialVelocity = np.array([0.0, 0.0, 1.0])
        velocity = np.tile(initialVelocity, (batchSize, 1))

        angleChanges = np.zeros(batchSize)
        energyChanges = np.zeros(batchSize)
        dt = 1 / 3

        # While at least one particle has energy > 0 and is inside voxel
        active = np.ones(batchSize, dtype=bool)

        while np.any(active):
            active_indices = np.where(active)[0]

            for i in active_indices:
                E = energy[i]
                if interp:
                    realAngle, realEnergy = sampleReverseCalculateInterpolation(
                        data, material, E, angleRange, energyRange, materialToIndex
                    )
                else:
                    realAngle, realEnergy = sampleReverseCalculate(
                        data, material, E, angleRange, energyRange, materialToIndex
                    )

                # Apply forward transform
                dEnergy, dAngle= variableChange(initialEnergy, realAngle, realEnergy)

                angleChanges[i] = dAngle
                energyChanges[i] = dEnergy
                energy[i] = realEnergy

                # Directional update
                phi = np.random.uniform(0, 2 * np.pi)
                theta = np.radians(realAngle)

                v = velocity[i]
                perp = np.cross(v, [0, 0, 1])
                norm = np.linalg.norm(perp)
                if norm < 1e-8:
                    perp = np.array([1, 0, 0])
                else:
                    perp /= norm

                w = (
                    np.cos(theta) * v +
                    np.sin(theta) * (np.cos(phi) * np.cross(perp, v) + np.sin(phi) * perp)
                )
                velocity[i] = w / np.linalg.norm(w)
                position[i] += velocity[i] * dt

                active[i] = (
                    realEnergy > 0 and checkIfInsideBigVoxel(position[i], bigVoxelSize)
                )

        return angleChanges.tolist(), energyChanges.tolist()

    except Exception as e:
        logging.exception(f"Batched simulation failed: {e}")
        return [], []


def simulateBatchWrapper(args):
    (
        batchSize, data, material, initialEnergy,
        angleRange, energyRange, materialToIndex,
        interp, bigVoxelSize
    ) = args

    return simulateBatchParticles(
        batchSize, data, material, initialEnergy,
        angleRange, energyRange, materialToIndex,
        interp, bigVoxelSize
    )
    
def runMultiprocessedBatchedSim(
    totalSamples, batchSize, numWorkers,
    data, material, initialEnergy,
    angleRange, energyRange, materialToIndex,
    interp, bigVoxelSize
):
    numBatches = (totalSamples + batchSize - 1) // batchSize

    argsList = [
        (
            min(batchSize, totalSamples - i * batchSize),  # size of current batch
            data, material, initialEnergy,
            angleRange, energyRange, materialToIndex,
            interp, bigVoxelSize
        )
        for i in range(numBatches)
    ]

    angleDistribution = []
    energyDistribution = []

    with Pool(processes=numWorkers) as pool:
        with tqdm(total=totalSamples, desc="Simulating (batched + mp)") as pbar:
            for result in pool.imap_unordered(simulateBatchWrapper, argsList):
                angles, energies = result
                angleDistribution.extend(angles)
                energyDistribution.extend(energies)
                pbar.update(len(angles)) 

    return angleDistribution, energyDistribution

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interp", action="store_true", help="Enable interpolation")
    args = parser.parse_args()
    logging.basicConfig(filename="simulation_errors.log", level=logging.ERROR)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    nProtonsTable = 10000
    npzPath = f'./Table/4DTable{nProtonsTable}.npz'
    samplingN = [10, # 100, 500, 1000, 5000, 10000, # 50000, 100000, 500000, 1000000
                ]
    numberOfBins = 100
    material = 'G4_WATER'
    initialEnergy = 200.0
    savePath = './Plots/'
    angleRange = (0, 70)
    energyRange = (-0.6, 0)
    bigVoxelSize = 33

    rawData = np.load(npzPath, allow_pickle=True)

    # Fully extract arrays here
    prob_table = rawData['probTable']
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
        startTime = time.time()

        batchSize = 1000
        numWorkers = cpu_count()

        angleDistribution, energyDistribution = runMultiprocessedBatchedSim(
            sampling, batchSize, numWorkers,
            data, material, initialEnergy,
            angleRange, energyRange, materialToIndex,
            args.interp, bigVoxelSize
        )

        endTime = time.time()
        timeOfHistories.append(endTime - startTime)
        print(f"Simulation time: {endTime - startTime:.2f} seconds")
        print()
        
    # Save results for time in CSV with two columns: samplingN and timeOfHistories
    with open('TimeOfHistoriesBatch.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        file.write("# SamplingN TimeOfHistories\n")
        for n, t in zip(samplingN, timeOfHistories):
            writer.writerow([n, t])
            
    # # Plot
    # plt.plot(samplingN, timeOfHistories, linestyle='-',marker = '.', color="black")      
    # plt.xlabel("Number of Histories")
    # plt.ylabel("Time (s)")
    # plt.tight_layout()  
    # plt.savefig(f"{savePath}HistoryTimePlotForEnergiesBatch.pdf")
    # # plt.show()
    # plt.close()

    # # Plot histograms
    # fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    # sns.histplot(energyDistribution, bins=numberOfBins, edgecolor="black", color='orange', kde=False, ax=axs[0]) 
    # axs[0].set_xlabel(r'$\frac{ln((E_i-E_f)/E_i)}{\sqrt{E_i}}$ (MeV$^{-1/2}$)')
    # axs[0].set_title('Final Energy distribution')
    # axs[0].set_yscale('log')
            
    # sns.histplot(angleDistribution, bins=numberOfBins, edgecolor="black", color='red', kde=False, ax=axs[1])
    # axs[1].set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
    # axs[1].set_title('Final Angles distribution')
    # axs[1].set_yscale('log')
    
    # plt.tight_layout()
    # plt.savefig(f'{savePath}MySimulationHistogramsBatch.pdf')
    # plt.close(fig)


    # hist1, xedges1, yedges1 = np.histogram2d(angleDistribution, energyDistribution, bins=numberOfBins)
    # finalProbabilities = hist1 / np.sum(hist1)

    # fig2, axs2 = plt.subplots(figsize=(8, 6))
    # h1 = axs2.pcolormesh(xedges1, yedges1, finalProbabilities.T, cmap='Reds', shading='auto')
    # fig2.colorbar(h1, ax=axs2, label='Probability')
    # axs2.set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
    # axs2.set_ylabel(r'$ln((E_i-E_f)/E_i)\sqrt{E_i}$ (MeV$^{-1/2}$)')

    # plt.tight_layout()
    # plt.savefig(f'{savePath}2DMySimulationHistogramsBatch.pdf')
    # plt.close(fig2) 