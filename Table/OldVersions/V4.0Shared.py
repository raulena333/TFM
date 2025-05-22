import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, shared_memory
import logging
import csv

samplerCache = {}

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

class HistogramSampler:
    def __init__(self, hist, angleRange, energyRange):
        self.hist = hist
        self.angleRange = angleRange
        self.energyRange = energyRange
        self.angleBins, self.energyBins = hist.shape

        self.flatHist = hist.flatten()
        self.cumsum = np.cumsum(self.flatHist)
        self.cumsum /= self.cumsum[-1]  # Normalize

        self.angleEdges = np.linspace(angleRange[0], angleRange[1], self.angleBins + 1)
        self.energyEdges = np.linspace(energyRange[0], energyRange[1], self.energyBins + 1)

        self.angleStep = self.angleEdges[1] - self.angleEdges[0]
        self.energyStep = self.energyEdges[1] - self.energyEdges[0]

    def sample(self):
        randValue = np.random.rand()
        idx = np.searchsorted(self.cumsum, randValue, side='right')
        angleIdx, energyIdx = np.unravel_index(idx, self.hist.shape)

        angle = self.angleEdges[angleIdx] + 0.5 * self.angleStep
        energy = self.energyEdges[energyIdx] + 0.5 * self.energyStep
        return angle, energy


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
    randValue = np.random.rand()
    idx = np.searchsorted(cumsum, randValue, side='right')

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


def sampleReverseCalculate(data, material, energy, angleRange, energyRange, materialToIndex):
    """
    Samples (angle, energy) from the joint histogram for a given material and energy level,
    then reverses the transformation to get real physical values.

    Parameters:
    - data (dict): Contains 'prob_table', 'energies', and 'materials'.
    - material (str): The material in which the particle exists.
    - energy (float): The initial energy of the proton.
    - angleRange (tuple): Range of transformed angles.
    - energyRange (tuple): Range of transformed energies.
    - materialToIndex (dict): Maps material names to their index in prob_table.

    Returns:
    - realAngle (float): Sampled physical angle.
    - realEnergy (float): Sampled physical energy.
    """
    
    if energy < 15.0:
        return 0.0, 0.0

    # Validate material
    if material not in materialToIndex:
        raise ValueError(f"Material '{material}' not found in data.")

    materialIdx = materialToIndex[material]
    availableEnergies = data['energies']
    
    # Find closest available energy index (to one decimal place)
    closestIdx = np.argmin(np.abs(np.round(availableEnergies, 1) - energy))

    # Fetch corresponding histogram
    hist = data['prob_table'][materialIdx, closestIdx]
    cacheKey = (materialIdx, closestIdx)
    
    if cacheKey not in samplerCache:
        samplerCache[cacheKey] = HistogramSampler(hist, angleRange, energyRange)

    sampler = samplerCache[cacheKey]
    sampledAngle, sampledEnergy = sampler.sample()
    realAngle, realEnergy = reverseVariableChange(energy, sampledAngle, sampledEnergy)

    return realAngle, realEnergy


def sampleReverseCalculateInterpolation(data, material, energy, angleRange, energyRange, materialToIndex):
    probTable = data['prob_table']
    energies = np.sort(data['energies'])

    if energy < 15.:
        return 0, 0
    if material not in materialToIndex:
        raise ValueError(f"Material '{material}' not found in data.")
        
    materialIdx = materialToIndex[material]
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

    if energyUp == energyLow:
        hist = probLow
    else:
        weight = (energy - energyLow) / (energyUp - energyLow)
        hist = (1 - weight) * probLow + weight * probHigh

    cache_key = (materialIdx, lowerIndex, upperIndex, round(weight, 4))  # tuple for unique key

    if cache_key not in samplerCache:
        samplerCache[cache_key] = HistogramSampler(hist, angleRange, energyRange)

    sampler = samplerCache[cache_key]
    angleSample, energySample = sampler.sample()
    realAngle, realEnergy = reverseVariableChange(energy, angleSample, energySample)

    return realAngle, realEnergy



def simulateBatchParticles(
    batchSize, data, material, initialEnergy,
    angleRange, energyRange, materialToIndex,
    interp, bigVoxelSize
):
    try:
        # Preallocate state arrays
        energy = np.full(batchSize, initialEnergy)
        
        # Pre-allocate position and velocity arrays that do not change per batch
        initialPos = np.array([0, 0, -bigVoxelSize], dtype=float)
        initialVel = np.array([0, 0, 1], dtype=float)
        position = np.broadcast_to(initialPos, (batchSize, 3)).copy()
        velocity = np.broadcast_to(initialVel, (batchSize, 3)).copy()

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
                    roundedEnergy = np.round(E, 1)
                    realAngle, realEnergy = sampleReverseCalculate(
                        data, material, roundedEnergy, angleRange, energyRange, materialToIndex
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
    

def simulateBatchParticles_worker(args):
    (
        shm_name, shape, dtype_str,
        batchSize, materials, energies,
        material, initialEnergy,
        angleRange, energyRange, materialToIndex,
        interp, bigVoxelSize
    ) = args

    # Attach to shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    prob_table = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=existing_shm.buf)

    # Reconstruct the data dictionary in each worker
    data = {
        'prob_table': prob_table,
        'materials': materials,
        'energies': energies
    }

    return simulateBatchParticles(
        batchSize, data, material, initialEnergy,
        angleRange, energyRange, materialToIndex,
        interp, bigVoxelSize
    )

    
def runMultiprocessedBatchedSim(
    totalSamples, batchSize, numWorkers,
    shm_prob_table, prob_table_shape, prob_table_dtype,
    data, material, initialEnergy,
    angleRange, energyRange, materialToIndex,
    interp, bigVoxelSize
):
    numBatches = (totalSamples + batchSize - 1) // batchSize

    argsList = [
        (
            shm_prob_table.name, prob_table_shape, prob_table_dtype.name,
            min(batchSize, totalSamples - i * batchSize),
            data['materials'], data['energies'],  # pass as raw lists/arrays
            material, initialEnergy,
            angleRange, energyRange, materialToIndex,
            interp, bigVoxelSize
        )
        for i in range(numBatches)
    ]

    angleDistribution = []
    energyDistribution = []

    with Pool(processes=numWorkers) as pool:
        with tqdm(total=totalSamples, desc="Simulating (batched)") as pbar:
            for result in pool.imap_unordered(simulateBatchParticles_worker, argsList):
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
    samplingN = [10, 100, 500, 1000, 5000, 10000, # 50000, 100000, 500000, 1000000
                 ]
    numberOfBins = 100 
    material = 'G4_WATER'
    initialEnergy = 200.0
    savePath = './Plots/'
    angleRange = (0, 70)
    energyRange = (-0.57, 0)
    bigVoxelSize = 33

    rawData = np.load(npzPath, allow_pickle=True)

    # Fully extract arrays here
    prob_table = rawData['probTable']
    materials = rawData['materials']
    energies = rawData['energies']
    
    # Create shared memory for the table
    shm_prob_table = shared_memory.SharedMemory(create=True, size=prob_table.nbytes)
    shm_prob_np = np.ndarray(prob_table.shape, dtype=prob_table.dtype, buffer=shm_prob_table.buf)
    np.copyto(shm_prob_np, prob_table)
    
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

        prob_table_shape = prob_table.shape
        prob_table_dtype = prob_table.dtype

        angleDistribution, energyDistribution = runMultiprocessedBatchedSim(
            sampling, batchSize, numWorkers,
            shm_prob_table, prob_table_shape, prob_table_dtype,
            data, material, initialEnergy,
            angleRange, energyRange, materialToIndex,
            args.interp, bigVoxelSize
        )

        endTime = time.time()
        timeOfHistories.append(endTime - startTime)
        print(f"Simulation time: {endTime - startTime:.2f} seconds")
        print()
        
    with open('TimeOfHistoriesShared.csv', 'w', newline='') as file:
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
    
    # Cleanup shared memory
    shm_prob_table.close()
    shm_prob_table.unlink()