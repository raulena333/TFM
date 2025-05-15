import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, shared_memory
import logging
import os
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

    def sample(self, size=1):
        randValues = np.random.rand(size)
        idxs = np.searchsorted(self.cumsum, randValues, side='right')
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

def sampleReverseCalculate_vectorized(data, material, energies, angleRange, energyRange, materialToIndex):
    materialIdx = materialToIndex[material]
    availableEnergies = data['energies']

    roundedEnergies = np.round(energies, 1)
    closestIndices = np.array([
        np.argmin(np.abs(availableEnergies - E)) for E in roundedEnergies
    ])

    sampledAngles = np.zeros_like(energies)
    sampledEnergies = np.zeros_like(energies)

    for idx in np.unique(closestIndices):
        cacheKey = (materialIdx, idx)
        if cacheKey not in samplerCache:
            hist = data['prob_table'][materialIdx, idx]
            samplerCache[cacheKey] = HistogramSampler(hist, angleRange, energyRange)

        sampler = samplerCache[cacheKey]

        indices = np.where(closestIndices == idx)[0]
        angles, energies_ = sampler.sample(size=indices.size)
        realAngles, realEnergies = reverseVariableChange(energies[indices], angles, energies_)
        
        sampledAngles[indices] = realAngles
        sampledEnergies[indices] = realEnergies

    return sampledAngles, sampledEnergies

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

def calculateEnergyDepositBin(position, bigVoxelSize, energyLoss, energyDepositedVector, voxelShapeBins=(200, 200, 200)):
    """
    Calculates and records the energy deposited in a voxel given the proton's position and energy loss.

    Parameters:
    - position (np.array or list): Current [x, y, z] position of the proton in mm.
    - bigVoxelSize (float): Half the full length of the cube (e.g., 33mm means cube spans from -33 to +33).
    - energyLoss (float): Amount of energy lost in this step.
    - energyDepositedVector (np.ndarray): 3D array to accumulate energy deposition per voxel.
    - voxelShapeBins (tuple): Number of bins (voxels) along each axis, e.g., (200, 200, 200).

    Returns:
    - energyDepositedVector (np.ndarray): Updated energy deposition array.
    """

    # Normalize position from [-bigVoxelSize, bigVoxelSize] to [0, 1]
    normalized = (np.array(position) + bigVoxelSize) / (2 * bigVoxelSize)
    
    # Convert normalized position to voxel indices
    voxelIndices = np.floor(normalized * voxelShapeBins).astype(int)

    # Clip indices to ensure they fall within bounds
    voxelIndices = np.clip(voxelIndices, [0, 0, 0], np.array(voxelShapeBins) - 1)

    # Deposit energy
    x, y, z = voxelIndices
    energyDepositedVector[x, y, z] += energyLoss

    return energyDepositedVector


def simulateBatchParticles_vectorized(
    batchSize, data, material, initialEnergy,
    angleRange, energyRange, materialToIndex,
    interp, bigVoxelSize,
    energyDeposited3D, voxelShapeBins
):
    energy = np.full(batchSize, initialEnergy)
    position = np.tile([0.0, 0.0, -bigVoxelSize], (batchSize, 1))
    velocity = np.tile([0.0, 0.0, 1.0], (batchSize, 1))
    dt = 1 / 3

    active = np.ones(batchSize, dtype=bool)
    angleChanges = np.zeros(batchSize)
    energyChanges = np.zeros(batchSize)

    while np.any(active):
        energyActive = energy[active]
        numActive = energyActive.size
        
        # Save current energy for delta calculation
        previousEnergy = energyActive.copy()

        if interp:
            realAngles = np.zeros(numActive)
            realEnergies = np.zeros(numActive)
            for i, E in enumerate(energyActive):
                realAngles[i], realEnergies[i] = sampleReverseCalculateInterpolation(
                    data, material, E, angleRange, energyRange, materialToIndex
                )
        else:
            realAngles = np.zeros(numActive)
            realEnergies = np.zeros(numActive)
            realAngles, realEnergies = sampleReverseCalculate_vectorized(
                data, material, energyActive, angleRange, energyRange, materialToIndex
            )

        dEnergies, dAngles = variableChange(initialEnergy, realAngles, realEnergies)
        energyLossPerStep = previousEnergy - realEnergies

        angleChanges[active] = dAngles
        energyChanges[active] = dEnergies
        energy[active] = realEnergies

        phi = np.random.uniform(0, 2 * np.pi, numActive)
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
        
        previousPosition = position[active].copy()
        position[active] += velocity[active] * dt

        for i, _ in enumerate(np.where(active)[0]):
            calculateEnergyDepositBin(
                previousPosition[i], bigVoxelSize, energyLossPerStep[i],
                energyDeposited3D, voxelShapeBins
            )

        insideVoxel = np.all(np.abs(position[active]) <= bigVoxelSize, axis=1)
        activeIndex = np.where(active)[0]
        active[activeIndex] = (realEnergies > 0) & insideVoxel

    finalAngles = np.degrees(np.arccos(velocity[:, 2])) * np.sqrt(initialEnergy)

    return finalAngles.tolist(), energyChanges.tolist(), energyDeposited3D


def simulateBatchParticles_worker(args):
    (
        shm_name, shape, dtype_str,
        shm_energy_name, energyDeposited3D_shape, energyDeposited3D_dtype,
        batchSize, materials, energies,
        material, initialEnergy,
        angleRange, energyRange, materialToIndex,
        interp, bigVoxelSize,
        voxelShapeBins
    ) = args

    np.random.seed(int(time.time() * 1000) % 2**32 + os.getpid())

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    prob_table = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=existing_shm.buf)

    shm_energy = shared_memory.SharedMemory(name=shm_energy_name)
    shared_energy_array = np.ndarray(energyDeposited3D_shape, dtype=np.dtype(energyDeposited3D_dtype), buffer=shm_energy.buf)

    data = {
        'prob_table': prob_table,
        'materials': materials,
        'energies': energies
    }

    return simulateBatchParticles_vectorized(
        batchSize, data, material, initialEnergy,
        angleRange, energyRange, materialToIndex,
        interp, bigVoxelSize,
        shared_energy_array, voxelShapeBins
    )
    
def runMultiprocessedBatchedSim(
    totalSamples, batchSize, numWorkers,
    shm_prob_table, prob_table_shape, prob_table_dtype,
    shm_energy, energyDeposited3D_shape, energyDeposited3D_dtype,
    data, material, initialEnergy,
    angleRange, energyRange, materialToIndex,
    interp, bigVoxelSize,
    voxelShapeBins
):
    numBatches = (totalSamples + batchSize - 1) // batchSize

    argsList = [
        (
            shm_prob_table.name, prob_table_shape, prob_table_dtype.name,
            shm_energy.name, energyDeposited3D_shape, energyDeposited3D_dtype.name,
            min(batchSize, totalSamples - i * batchSize),
            data['materials'], data['energies'],
            material, initialEnergy,
            angleRange, energyRange, materialToIndex,
            interp, bigVoxelSize,
            voxelShapeBins
        )
        for i in range(numBatches)
    ]

    angleDistribution = []
    energyDistribution = []

    with Pool(processes=numWorkers) as pool:
        with tqdm(total=totalSamples, desc="Simulating (batched)") as pbar:
            for result in pool.imap_unordered(simulateBatchParticles_worker, argsList):
                angles, energies, _ = result
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

    fileforEnergyDeposit = "./EnergyAtBoxByBinsMySimulation.csv"
    
    nProtonsTable = 100000
    npzPath = f'./Table/4DTable{nProtonsTable}.npz'
    samplingN = [10000000]
    numberOfBins = 100 
    material = 'G4_WATER'
    initialEnergy = 200.0
    savePath = './Plots/'
    timingPath = './Timing/'
    angleRange = (0, 70)
    energyRange = (-0.57, 0)
    bigVoxelSize = 33
    voxelShapeBins = (200, 200, 200)

    # Make sure the timing directory exists
    os.makedirs(timingPath, exist_ok=True)
    
    rawData = np.load(npzPath, allow_pickle=True)
    # Fully extract arrays here
    prob_table = rawData['prob_table']
    materials = rawData['materials']
    energies = rawData['energies']
    rawData.close()
    
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
    materialToIndex = {mat: idx for idx, mat in enumerate(data['materials'])}
    
    energyDeposited3D = np.zeros(voxelShapeBins, dtype=np.float64)
    shm_energy = shared_memory.SharedMemory(create=True, size=energyDeposited3D.nbytes)
    shared_energy_array = np.ndarray(energyDeposited3D.shape, dtype=energyDeposited3D.dtype, buffer=shm_energy.buf)
    np.copyto(shared_energy_array, energyDeposited3D)
    
    angleDistribution = []
    energyDistribution = []
    timeOfHistories = []
    
    try:
        for sampling in samplingN:
            startTime = time.time()

            batchSize = 10
            numWorkers = cpu_count()

            prob_table_shape = prob_table.shape
            prob_table_dtype = prob_table.dtype

            angleDistribution, energyDistribution = runMultiprocessedBatchedSim(
                sampling, batchSize, numWorkers,
                shm_prob_table, prob_table_shape, prob_table_dtype,
                shm_energy, energyDeposited3D.shape, energyDeposited3D.dtype,
                data, material, initialEnergy,
                angleRange, energyRange, materialToIndex,
                args.interp, bigVoxelSize,
                voxelShapeBins
            )
            
            energyDepositedArray = np.ndarray(
                energyDeposited3D.shape,
                dtype=energyDeposited3D.dtype,
                buffer=shm_energy.buf
            )   

            endTime = time.time()
            timeOfHistories.append(endTime - startTime)
            print(f"Simulation time: {endTime - startTime:.2f} seconds")
            print()

    # Cleanup shared memory     
    finally:
        with open(fileforEnergyDeposit, 'w', newline='') as file:
            file.write(
                "# Simulation Version: 4.\n"
                "# Results for scorer: EnergyDeposit\n"
                "# Scored in component: Box\n"
                "# EnergyDeposit (MeV): Sum\n"
                f"# X in {voxelShapeBins[0]} bins of 0.1 cm\n"
                f"# Y in {voxelShapeBins[1]} bins of 0.1 cm\n"
                f"# Z in {voxelShapeBins[2]} bins of 0.1 cm\n"
            )
            writer = csv.writer(file, delimiter=' ')
            for x in range(energyDepositedArray.shape[0]):
                for y in range(energyDepositedArray.shape[1]):
                    for z in range(energyDepositedArray.shape[2]):
                        value = energyDepositedArray[x, y, z]
                        if value > 0:
                            writer.writerow([x, y, z, f"{value:.6f}"])
                            
        shm_prob_table.close()
        shm_prob_table.unlink()
        shm_energy.close()
        shm_energy.unlink()