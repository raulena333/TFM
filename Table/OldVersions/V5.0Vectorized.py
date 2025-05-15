import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, shared_memory
import logging

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
        idxs = np.searchsorted(self.cumsum, randValue, side='right')
        angleIdxs, energyIdxs = np.unravel_index(idxs, self.hist.shape)

        angles = self.angleEdges[angleIdxs] + 0.5 * self.angleStep
        energies = self.energyEdges[energyIdxs] + 0.5 * self.energyStep
        return angles, energies

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

def simulateBatchParticles_vectorized(
    batchSize, data, material, initialEnergy,
    angleRange, energyRange, materialToIndex,
    interp, bigVoxelSize
):
    energy = np.full(batchSize, initialEnergy)
    position = np.tile([0.0, 0.0, -bigVoxelSize], (batchSize, 1))
    velocity = np.tile([0.0, 0.0, 1.0], (batchSize, 1))
    dt = 1 / 3

    active = np.ones(batchSize, dtype=bool)
    angleChanges = np.zeros(batchSize)
    energyChanges = np.zeros(batchSize)

    while np.any(active):
        E_active = energy[active]
        num_active = E_active.size

        if interp:
            realAngles = np.zeros(num_active)
            realEnergies = np.zeros(num_active)
            for i, E in enumerate(E_active):
                realAngles[i], realEnergies[i] = sampleReverseCalculateInterpolation(
                    data, material, E, angleRange, energyRange, materialToIndex
                )
        else:
            roundedEnergies = np.round(E_active, 1)
            realAngles = np.zeros(num_active)
            realEnergies = np.zeros(num_active)
            for i, E in enumerate(roundedEnergies):
                realAngles[i], realEnergies[i] = sampleReverseCalculate(
                    data, material, E, angleRange, energyRange, materialToIndex
                )

        dEnergies, dAngles = variableChange(initialEnergy, realAngles, realEnergies)

        angleChanges[active] = dAngles
        energyChanges[active] = dEnergies
        energy[active] = realEnergies

        phi = np.random.uniform(0, 2 * np.pi, num_active)
        theta = np.radians(realAngles)

        v = velocity[active]
        perp = np.cross(v, np.array([0, 0, 1]))
        norm = np.linalg.norm(perp, axis=1, keepdims=True)
        perp[norm[:, 0] < 1e-8] = np.array([1, 0, 0])
        perp = perp / np.where(norm == 0, 1, norm)

        cross_perp_v = np.cross(perp, v)
        w = (
            np.cos(theta)[:, np.newaxis] * v +
            np.sin(theta)[:, np.newaxis] * (
                np.cos(phi)[:, np.newaxis] * cross_perp_v +
                np.sin(phi)[:, np.newaxis] * perp
            )
        )
        velocity[active] = w / np.linalg.norm(w, axis=1, keepdims=True)
        position[active] += velocity[active] * dt

        inside_voxel = np.all(np.abs(position[active]) <= bigVoxelSize, axis=1)
        active_indices = np.where(active)[0]
        active[active_indices] = (realEnergies > 0) & inside_voxel

    return angleChanges.tolist(), energyChanges.tolist()
    

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

    return simulateBatchParticles_vectorized(
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

    nProtonsTable = 100000
    npzPath = f'./Table/4DTable{nProtonsTable}.npz'
    samplingN = [10, 100, 500, 1000, # 5000, 10000, 50000, 100000, 500000, 1000000
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
    prob_table = rawData['prob_table']
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
     
    try:
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
            
    # Cleanup shared memory     
    finally:
        shm_prob_table.close()
        shm_prob_table.unlink()
        
    # Plot
    plt.plot(samplingN, timeOfHistories, linestyle='-',marker = '.', color="black")      
    plt.xlabel("Number of Histories")
    plt.ylabel("Time (s)")
    plt.tight_layout()  
    plt.savefig(f"{savePath}HistoryTimePlotForEnergiesBatch.pdf")
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
    plt.savefig(f'{savePath}MySimulationHistogramsBatchSharedVector.pdf')
    plt.close(fig)


    hist1, xedges1, yedges1 = np.histogram2d(angleDistribution, energyDistribution, bins=numberOfBins)
    finalProbabilities = hist1 / np.sum(hist1)

    fig2, axs2 = plt.subplots(figsize=(8, 6))
    h1 = axs2.pcolormesh(xedges1, yedges1, finalProbabilities.T, cmap='Reds', shading='auto')
    fig2.colorbar(h1, ax=axs2, label='Probability')
    axs2.set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
    axs2.set_ylabel(r'$ln((E_i-E_f)/E_i)\sqrt{E_i}$ (MeV$^{-1/2}$)')

    plt.tight_layout()
    plt.savefig(f'{savePath}2DMySimulationHistogramsBatch.pdf')
    plt.close(fig2) 