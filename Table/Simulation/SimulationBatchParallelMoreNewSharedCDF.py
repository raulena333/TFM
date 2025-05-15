import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import time
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, shared_memory
import os
import csv
from numba import prange
import numba
from joblib import Parallel, delayed
import psutil
import faulthandler
import seaborn as sns
import pandas as pd
from typing import Tuple
from pathlib import Path

samplerCache = {}
faulthandler.enable()

# Plot styling
params = {
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'legend.fontsize': 14
}
pylab.rcParams.update(params)

# --------------- TRANSFORMATION VARIABLE ---------------
# This class represents a binning configuration for angles and energies, used for sampling for the transformation variable.
# Initializes the binning configuration with the given angle and energy ranges, and number of bins.
class BinningConfig:
    def __init__(self, angleRange : tuple, energyRange : tuple, angleBins : int, energyBins : int):
        self.angleRange = angleRange
        self.energyRange = energyRange
        self.angleBins = angleBins
        self.energyBins = energyBins

        # Create unified binEdges array: shape (2, maxBins)
        maxBins = max(angleBins + 1, energyBins + 1)
        self.binEdges = np.zeros((2, maxBins), dtype=np.float32)

        # Compute bin edges
        self.binEdges[0, :angleBins + 1] = np.linspace(angleRange[0], angleRange[1], angleBins + 1, dtype=np.float32)
        self.binEdges[1, :energyBins + 1] = np.linspace(energyRange[0], energyRange[1], energyBins + 1, dtype=np.float32)

        self.angleStep = self.binEdges[0, 1] - self.binEdges[0, 0]
        self.energyStep = self.binEdges[1, 1] - self.binEdges[1, 0]
        
    # Return the precomputed values
    def getBinningValues(self) -> Tuple[np.ndarray, float, float]:
        return self.binEdges, self.angleStep, self.energyStep

# --------------- TRANSFORMATION VARIABLE ---------------
# This class represents a histogram sampler for sampling angles and energies based on a given histogram.
# It initializes with a histogram, calculates the cumulative distribution function (CDF), and provides a method to sample angles and energies.
class HistogramSampler:

    def __init__(self, hist : np.ndarray, rng=None):
        self.hist = hist
        self.angleBins, self.energyBins = hist.shape
        self.rng = rng or np.random.default_rng()

        self.flatHist = hist.flatten()
        self.cumsum = np.cumsum(self.flatHist)
        self.cumsum /= self.cumsum[-1] 
        
    def sample(self, size=1) -> Tuple[np.ndarray, np.ndarray]:
        randValues = self.rng.random(size)
        idxs = np.searchsorted(self.cumsum, randValues, side='right')
        angleIdxs, energyIdxs = np.unravel_index(idxs, (self.angleBins, self.energyBins))

        # Use global binning config
        angles = binningConfig.angleEdges[angleIdxs] + 0.5 * binningConfig.angleStep
        energies = binningConfig.energyEdges[energyIdxs] + 0.5 * binningConfig.energyStep

        return angles, energies

@numba.jit(parallel=True)
def buildCdfsFromProbTable(probTable):
    numMaterials, numEnergies, angleBins, energyBins = probTable.shape
    totalBins = angleBins * energyBins

    cdfs = np.empty((numMaterials, numEnergies, totalBins), dtype=np.float32)

    for m in prange(numMaterials):  # parallel over materials
        for e in range(numEnergies):
            flatProbTable = probTable[m, e].ravel()  # flatten 2D bin table
            cdf = np.empty_like(flatProbTable)
            total = 0.0
            for i in range(totalBins):  # compute cumulative sum manually
                total += flatProbTable[i]
                cdf[i] = total
            norm = cdf[-1]
            for i in range(totalBins):  # normalize manually
                cdf[i] /= norm
                cdfs[m, e, i] = cdf[i]

    return cdfs

# --------------- TRANSFORMATION VARIABLE --------------
def prebuildSamplers(data : dict, angleRange : tuple, energyRange : tuple, materialToIndex : dict):
    global samplerCache, binningConfig

    # Create global binning config
    angleBins, energyBins = data['probTable'].shape[2:]
    binningConfig = BinningConfig(angleRange, energyRange, angleBins, energyBins)

    for materialIdx in materialToIndex.values():
        for energyIdx in range(len(data['energies'])):
            cacheKey = (materialIdx, energyIdx)
            if cacheKey not in samplerCache:
                hist = data['probTable'][materialIdx, energyIdx]
                samplerCache[cacheKey] = HistogramSampler(hist)
                
# --------------- TRANSFORMATION VARIABLE --------------
@numba.jit(nopython=True, inline = 'always') 
def reverseVariableChangeTransform(initialEnergies, angles, energies):
    realAngles = angles / np.sqrt(initialEnergies)
    realEnergies = initialEnergies * (1.0 - np.exp(energies * np.sqrt(initialEnergies)))
    return realAngles, realEnergies                
                
# --------------- TRANSFORMATION VARIABLE --------------
def sampleReverseCalculateInterpolation(data : dict, material : str, energy : float, 
        angleRange : tuple,  energyRange : tuple,  materialToIndex : dict):
    probTable = data['probTable']
    energies = np.sort(data['energies'])

    if energy < 9.:
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
    realAngle, realEnergy = reverseVariableChangeTransform(energy, angleSample, energySample)

    return realAngle, realEnergy   

# --------------- TRANSFORMATION VARIABLE ---------------       
def sampleFromCDFVectorizedTransformation(data: dict, material: str, energies: np.ndarray, materialToIndex: dict, cdfs: np.ndarray,
        binEdges: np.ndarray, angleStep : float, energyStep : float):
    # Get the index of the material from the lookup dictionary
    materialIdx = materialToIndex[material]
    availableEnergies = data['energies']

    # Clip energies to the valid range
    minEnergy = np.min(availableEnergies)
    maxEnergy = np.max(availableEnergies)
    roundedEnergies = np.clip(np.round(energies, 1), minEnergy, maxEnergy)

    # For each input energy, find the index of the closest available energy
    # closestIndices = np.array([
    #     np.argmin(np.abs(availableEnergies - E)) for E in roundedEnergies
    # ])
    
    # availableEnergies is descending, so reverse it for searchsorted
    reversedEnergies = availableEnergies[::-1]

    # Perform search in ascending order
    insertPos = np.searchsorted(reversedEnergies, roundedEnergies, side='left')

    # Clamp insertPos within valid range
    insertPos = np.clip(insertPos, 1, len(reversedEnergies) - 1)

    # Compare to both neighbors in reversed array
    left = reversedEnergies[insertPos - 1]
    right = reversedEnergies[insertPos]
    chooseLeft = np.abs(roundedEnergies - left) < np.abs(roundedEnergies - right)

    # Compute correct index back in original (descending) energy array
    closestIndices = len(availableEnergies) - 1 - np.where(chooseLeft, insertPos - 1, insertPos)

    sampledAngles = np.zeros_like(energies, dtype=np.float32)
    sampledEnergies = np.zeros_like(energies, dtype=np.float32)

    # Initialize binning configuration for interpolation of CDF samples
    angleBins, energyBins = data['probTable'].shape[2:]
    
    # Use global binEdges shared across all materials and energies
    angleEdges = binEdges[0, :angleBins + 1]
    energyEdges = binEdges[1, :energyBins + 1]
    
    # Sample in batches for each unique energy index
    uniqueEnergyIndices = np.unique(closestIndices)
    for energyIdx in uniqueEnergyIndices:
        sampleIndices = np.where(closestIndices == energyIdx)[0].astype(np.int32)
        randValues = np.random.random(size=sampleIndices.size).astype(np.float32)

        sampleCDFForEnergyGroup(
            materialIdx, energyIdx, sampleIndices, randValues,
            cdfs, angleBins, energyBins, angleEdges, energyEdges,
            angleStep, energyStep, sampledAngles, sampledEnergies
        )

    # Apply transformation back to physical space
    realAngles, realEnergies = reverseVariableChangeTransform(energies, sampledAngles, sampledEnergies)
    
    # For input energies below the minimum threshold, zero out the output
    mask = energies < minEnergy
    realAngles[mask] = 0
    realEnergies[mask] = 0

    return realAngles, realEnergies


# --------------- NORMALIZATION VARIABLE --------------
def buildCdfsAndCompactBins(data: dict) -> Tuple[np.ndarray, np.ndarray]:
    probTable = data['probTable']
    thetaMax = data['thetaMax']
    thetaMin = data['thetaMin']
    energyMin = data['energyMin']
    energyMax = data['energyMax']
    
    numMaterials, numEnergies, angleBins, energyBins = probTable.shape
    totalBins = angleBins * energyBins
    maxBinCount = max(angleBins + 1, energyBins + 1)

    cdfs = np.zeros((numMaterials, numEnergies, totalBins), dtype=np.float32)
    # Unified bin edges: [m, e, 0] = angleEdges, [m, e, 1] = energyEdges
    binEdges = np.zeros((numMaterials, numEnergies, 2, maxBinCount), dtype=np.float32)
    
    # Precompute normalized linspaces for re-use
    normAngle = np.linspace(0, 1, angleBins + 1, dtype=np.float32)
    normEnergy = np.linspace(0, 1, energyBins + 1, dtype=np.float32)

    for m in range(numMaterials):
        for e in range(numEnergies):
            hist = probTable[m, e].reshape(-1)
            total = hist.sum(dtype=np.float32)

            if total > 0:
                cdfs[m, e] = np.cumsum(hist, dtype=np.float32) / total

            thetaRange = thetaMax[m, e] - thetaMin[m, e]
            energyRange = energyMax[m, e] - energyMin[m, e]
            
            # Slicing, although the bins are the same size for all materials and energies
            binEdges[m, e, 0, :angleBins + 1] = thetaMin[m, e] + thetaRange * normAngle
            binEdges[m, e, 1, :energyBins + 1] = energyMin[m, e] + energyRange * normEnergy

    return cdfs, binEdges

# --------------- NORMALIZATION VARIABLE --------------
@numba.jit(nopython=True, inline = 'always') 
def reverseVariableChangeNormalized(normalizedAngle, normalizedEnergy, thetaMax, thetaMin, energyMin, energyMax):
    realAngle = normalizedAngle * (thetaMax - thetaMin) + thetaMin
    realEnergy = energyMin + normalizedEnergy * (energyMax - energyMin)
    return realAngle, realEnergy

# --------------- NORMALIZATION VARIABLE -------------
def sampleFromCDFVectorizedNormalization(
    data: dict, 
    material: str, 
    energies: np.ndarray, 
    materialToIndex: dict,
    cdfs: np.ndarray, 
    binEdges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    materialIdx = materialToIndex[material]
    availableEnergies = data['energies']
    angleBins, energyBins = data['probTable'].shape[2:]

    minEnergy = np.min(availableEnergies)
    maxEnergy = np.max(availableEnergies)
    roundedEnergies = np.clip(np.round(energies, 1), minEnergy, maxEnergy)

    # Interpolation setup
    reversedEnergies = availableEnergies[::-1]
    insertPos = np.searchsorted(reversedEnergies, roundedEnergies, side='left')
    insertPos = np.clip(insertPos, 1, len(reversedEnergies) - 1)

    upperIndices = len(availableEnergies) - 1 - (insertPos - 1)
    lowerIndices = np.clip(upperIndices - 1, 0, len(availableEnergies) - 1)
    upperIndices = np.clip(upperIndices, 0, len(availableEnergies) - 1)

    lowerEnergy = availableEnergies[lowerIndices]
    upperEnergy = availableEnergies[upperIndices]

    # Interpolation weights, only applied where energy >= minEnergy
    weights = np.where(
        upperEnergy != lowerEnergy, 
        (roundedEnergies - lowerEnergy) / (upperEnergy - lowerEnergy), 
        0.0
    ).astype(np.float32)

    rand1 = np.random.random(size=energies.shape).astype(np.float32)
    rand2 = np.random.random(size=energies.shape).astype(np.float32)

    sampledAnglesLower = np.zeros_like(energies, dtype=np.float32)
    sampledEnergiesLower = np.zeros_like(energies, dtype=np.float32)
    sampledAnglesUpper = np.zeros_like(energies, dtype=np.float32)
    sampledEnergiesUpper = np.zeros_like(energies, dtype=np.float32)

    for (indices, rand, outAngles, outEnergies) in [
        (lowerIndices, rand1, sampledAnglesLower, sampledEnergiesLower),
        (upperIndices, rand2, sampledAnglesUpper, sampledEnergiesUpper),
    ]:
        uniqueEnergyIndices = np.unique(indices)
        for energyIdx in uniqueEnergyIndices:
            idxs = np.where(indices == energyIdx)[0].astype(np.int32)

            angleEdges = binEdges[materialIdx, energyIdx, 0, :angleBins + 1]
            energyEdges = binEdges[materialIdx, energyIdx, 1, :energyBins + 1]

            if angleEdges[0] == angleEdges[-1] or energyEdges[0] == energyEdges[-1]:
                outAngles[idxs] = 0.0
                outEnergies[idxs] = 0.0
                continue

            angleStep = angleEdges[1] - angleEdges[0]
            energyStep = energyEdges[1] - energyEdges[0]

            sampleCDFForEnergyGroup(
                materialIdx, energyIdx, idxs, rand[idxs],
                cdfs, angleBins, energyBins, angleEdges, energyEdges,
                angleStep, energyStep, outAngles, outEnergies
            )

    # Final interpolation only for energies >= minEnergy
    interpolateMask = energies >= minEnergy
    sampledAngles = np.where(
        interpolateMask,
        (1.0 - weights) * sampledAnglesLower + weights * sampledAnglesUpper,
        sampledAnglesLower
    )
    sampledEnergies = np.where(
        interpolateMask,
        (1.0 - weights) * sampledEnergiesLower + weights * sampledEnergiesUpper,
        sampledEnergiesLower
    )

    return sampledAngles, sampledEnergies


# --------------- COMMON FUNCTIONS ---------------
@numba.jit(nopython=True, inline = 'always')
def sampleCDFForEnergyGroup(materialIdx, energyIdx, sampleIndices, randValues, cdfs,
            angleBins, energyBins, angleEdges, energyEdges, angleStep, energyStep,
            sampledAngles, sampledEnergies
):
    cdf = cdfs[materialIdx, energyIdx]

    for i in range(sampleIndices.size):
        idx = sampleIndices[i]
        r = randValues[i]

        flatIdx = np.searchsorted(cdf, r)
        if flatIdx >= cdf.size:
            flatIdx = cdf.size - 1

        angleIdx = flatIdx // energyBins
        energyIdxLocal = flatIdx % angleBins

        angle = angleEdges[angleIdx] + 0.5 * angleStep
        energy = energyEdges[energyIdxLocal] + 0.5 * energyStep

        sampledAngles[idx] = angle
        sampledEnergies[idx] = energy
        
# --------------- COMMON FUNCTIONS ---------------
def createPhysicalSpace(bigVoxel : tuple,  voxelShapeBins : tuple, 
        dt = 1 / 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xRange = np.linspace(-bigVoxel[0] / dt, bigVoxel[0] / dt, voxelShapeBins[0]) 
    yRange = np.linspace(-bigVoxel[1] / dt, bigVoxel[1] / dt, voxelShapeBins[1])
    zRange = np.linspace(-bigVoxel[2] / dt, bigVoxel[2] / dt, voxelShapeBins[2])
    
    return xRange, yRange, zRange

# --------------- COMMON FUNCTIONS ---------------
@numba.jit(nopython=True)
def calculateEnergyDepositBinBatch(positions : np.ndarray, physicalSize : tuple, energyLosses : np.ndarray, 
            energyDepositedVector : np.ndarray, voxelShapeBins : tuple):
    n = positions.shape[0]
    sizeX, sizeY, sizeZ = physicalSize[0], physicalSize[1], physicalSize[2]
    binsX, binsY, binsZ = voxelShapeBins

    for i in prange(n):
        x = (positions[i, 0] + sizeX) / (2 * sizeX)
        y = (positions[i, 1] + sizeY) / (2 * sizeY)
        z = (positions[i, 2] + sizeZ) / (2 * sizeZ)

        ix = int(x * binsX)
        iy = int(y * binsY)
        iz = int(z * binsZ)

        if ix < 0:
            ix = 0
        elif ix >= binsX:
            ix = binsX - 1

        if iy < 0:
            iy = 0
        elif iy >= binsY:
            iy = binsY - 1

        if iz < 0:
            iz = 0
        elif iz >= binsZ:
            iz = binsZ - 1

        energyDepositedVector[ix, iy, iz] += energyLosses[i]
    
        # print(f"Energy deposited in voxel bin {ix}, {iy}, {iz}:", energyDepositedVector[ix, iy, iz])
        
    return energyDepositedVector

# --------------- COMMON FUNCTIONS ---------------
def sampleFromCDF(
    data, material, energies, materialToIndex, cdfs, binEdges, method="transformation",
    angleStep=None, energyStep=None
):
    if method == "transformation":
        return sampleFromCDFVectorizedTransformation(
            data=data,
            material=material,
            energies=energies,
            materialToIndex=materialToIndex,
            cdfs=cdfs,
            binEdges=binEdges,
            angleStep=angleStep,
            energyStep=energyStep
        )
    elif method == "normalization":
        if binEdges is None:
            raise ValueError("Bin edges must be provided for 'normalization' method.")
        return sampleFromCDFVectorizedNormalization(
            data=data,
            material=material,
            energies=energies,
            materialToIndex=materialToIndex,
            cdfs=cdfs,
            binEdges=binEdges
        )
    else:
        raise ValueError(f"Unknown sampling method: {method}")
        
# --------------- COMMON FUNCTIONS ---------------
def simulateBatchParticlesVectorized(
    batchSize, data, material, initialEnergy,
    materialToIndex, bigVoxelSize, 
    energyDepositedVector, cdfs, 
    binEdges, method = 'transformation',
    angleStep = None, energyStep = None
):
    smallStep = 1e-3
    dt = 1 / 3 + smallStep
    energy = np.full(batchSize, initialEnergy)
    position = np.tile([0.0, 0.0, -bigVoxelSize[2]], (batchSize, 1))
    velocity = np.tile([0.0, 0.0, 1.0], (batchSize, 1))   

    active = np.ones(batchSize, dtype=bool)
    zAxis = np.array([0.0, 0.0, 1.0])

    # print("Initial energy:", energy, "Initial position:", position)
    
    while np.any(active):
        energyActive = energy[active]
        n = energyActive.shape[0]
        
        realAngles, realEnergies = sampleFromCDF(data=data, material=material, energies=energyActive, 
                materialToIndex=materialToIndex, cdfs=cdfs, binEdges=binEdges, method=method, 
                angleStep=angleStep, energyStep=energyStep)

        energyLossPerStep = energyActive - realEnergies
        energy[active] = realEnergies
    
        # Print interesting data
        #print(f"EnergyActive: {energyActive}, SampleEnergies: {realEnergies}, energyLossPerStep: {energyLossPerStep}")
        #print(f"SampleAngles: {realAngles}, Position: {position[active]}")
        calculateEnergyDepositBinBatch(
            position[active], bigVoxelSize, energyLossPerStep,
            energyDepositedVector, energyDepositedVector.shape
        )
         
        # === Update velocity based on scattering ===
        phi = np.random.uniform(0, 2 * np.pi, n)
        theta = np.radians(realAngles)

        v = velocity[active]
        perp = np.cross(v, zAxis)
        norms = np.linalg.norm(perp, axis=1, keepdims=True)
        
        # Handle edge case where the cross product is zero
        perp[norms[:, 0] < 1e-8] = np.array([1.0, 0.0, 0.0])
        perp[~(norms[:, 0] < 1e-8)] /= norms[~(norms[:, 0] < 1e-8)]

        crossPerpV = np.cross(perp, v)
        cosTheta = np.cos(theta)[:, np.newaxis]
        sinTheta = np.sin(theta)[:, np.newaxis]
        cosPhi = np.cos(phi)[:, np.newaxis]
        sinPhi = np.sin(phi)[:, np.newaxis]

        w = cosTheta * v + sinTheta * (cosPhi * crossPerpV + sinPhi * perp)
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        velocity[active] = w
        
        # Update position
        position[active] += velocity[active] * dt

        # Check if particles are still active
        withinBounds = np.all(
            (position[active] >= -bigVoxelSize) & (position[active] <= bigVoxelSize),
            axis=1
        )
        realEnergiesValid = realEnergies > 0
        active[active] = realEnergiesValid & withinBounds
        
# --------------- COMMON FUNCTIONS ---------------
def simulateBatchParticlesWorker(args):
    method = args[17] 
    
    # Common extraction for all methods
    (
        shm_name, shape, dtype_str,
        shm_cdfs_name, cdfs_shape, cdfs_dtype_str,
        shm_energy_deposited_name, energy_deposited_shape, energy_deposited_dtype_str,
        batchSize, materials, energies,
        material, initialEnergy,
        materialToIndex, bigVoxelSize,
        seed, _  # method is already extracted
    ) = args[:18]

    # Default to None (for both methods)
    binEdges = angleStep = energyStep = None
    shm_bin_edges_name = bin_edges_shape = bin_edges_dtype_str = None

    # Handle method-specific arguments
    if method == 'normalization':
        (
            shm_bin_edges_name, bin_edges_shape, bin_edges_dtype_str
        ) = args[18:]

    elif method == 'transformation':
        (
            binEdges, angleStep, energyStep
        ) = args[18:]

    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    # Seed for reproducibility
    np.random.seed(seed)
    
    # Attach to shared memory for prob_table
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    probTable = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=existing_shm.buf)
    
    # Attach to shared memory for cdfs
    existing_shm_cdfs = shared_memory.SharedMemory(name=shm_cdfs_name)
    cdfs = np.ndarray(cdfs_shape, dtype=np.dtype(cdfs_dtype_str), buffer=existing_shm_cdfs.buf)

    # Attach to shared memory for energyDepositedVector
    existing_shm_energy_deposited = shared_memory.SharedMemory(name=shm_energy_deposited_name)
    energyDepositedVector = np.ndarray(energy_deposited_shape, dtype=np.dtype(energy_deposited_dtype_str), buffer=existing_shm_energy_deposited.buf)

    # Reconstruct the data dictionary in each worker
    data = {
        'probTable': probTable,
        'materials': materials,
        'energies': energies
    }
    
    if method == 'normalization':
        # Attach to shared memory for binEdges
        existing_shm_bin_edges = shared_memory.SharedMemory(name=shm_bin_edges_name)
        binEdges = np.ndarray(bin_edges_shape, dtype=np.dtype(bin_edges_dtype_str), buffer=existing_shm_bin_edges.buf)
        
    return simulateBatchParticlesVectorized(batchSize=batchSize, data=data, material=material, initialEnergy=initialEnergy,
            materialToIndex=materialToIndex, bigVoxelSize=bigVoxelSize, energyDepositedVector=energyDepositedVector, cdfs=cdfs,
            binEdges=binEdges, method=method, angleStep=angleStep, energyStep=energyStep)
    

# --------------- COMMON FUNCTIONS ---------------
def simulateBatch(args):
    return simulateBatchParticlesWorker(args)

# --------------- COMMON FUNCTIONS ---------------
def runMultiprocessedBatchedSim(
    totalSamples, batchSize, numWorkers,
    shm_prob_table, prob_table_shape, prob_table_dtype,
    shm_cdfs, cdfs_shape, cdfs_dtype,
    shm_energy_deposited, energy_deposited_shape, energy_deposited_dtype,
    data, material, initialEnergy, materialToIndex, 
    bigVoxelSize, method = 'transformation',
    #Optional for transformation
    binEdges = None, angleStep = None, energyStep = None,
    # Optional shared memory for normalization
    shm_bin_edges = None, bin_edges_shape = None, bin_edges_dtype = None,
):
    
    # Safety checks
    if method == 'normalization':
        assert shm_bin_edges is not None, "shm_bin_edges is required for normalization method."
        assert bin_edges_shape is not None, "bin_edges_shape is required for normalization method."
        assert bin_edges_dtype is not None, "bin_edges_dtype is required for normalization method."

    elif method == 'transformation':
        assert binEdges is not None, "binEdges is required for transformation method."
        assert angleStep is not None, "angleStep is required for transformation method."
        assert energyStep is not None, "energyStep is required for transformation method."

    else:
        raise ValueError(f"Unsupported sampling method: {method}")
    
    # Number of batches to process
    numBatches = (totalSamples + batchSize - 1) // batchSize
    argsList = []
    
    baseSeed = 189853376
    seedSequence = np.random.SeedSequence(baseSeed)
    childSeeds = seedSequence.spawn(numBatches)
    
    # Create a list of arguments for each worker with chunked tasks
    for i in range(numBatches):
        currentSeed = childSeeds[i].generate_state(1)[0]
        
        args = [
            shm_prob_table.name, prob_table_shape, prob_table_dtype.name,
            shm_cdfs.name, cdfs_shape, cdfs_dtype.name,
            shm_energy_deposited.name, energy_deposited_shape, energy_deposited_dtype.name,
            min(batchSize, totalSamples - i * batchSize),
            data['materials'], data['energies'], 
            material, initialEnergy, materialToIndex,
            bigVoxelSize, 
            currentSeed, # seed for each batch, reproducible
            method,
        ]

        if method == 'normalization':
            args.extend([
                shm_bin_edges.name, bin_edges_shape, bin_edges_dtype.name
            ])
        elif method == 'transformation':
            args.extend([
                binEdges, angleStep, energyStep      
            ])

        argsList.append(tuple(args))
            
    # chunkSize = max(1, len(argsList) // (numWorkers * 4)) 
    chunkSize = 1
    with Pool(processes=numWorkers) as pool:
        list(tqdm(pool.imap_unordered(simulateBatchParticlesWorker, argsList, chunksize=chunkSize), total=len(argsList)))
        
# --------------- COMMON FUNCTIONS ---------------
def plotSamplingDistribution(data : dict, material : str, fixedEnergyIdx : int, materialToIndex : dict, cdfs : np.ndarray, 
            method='transformation', N=10000, binEdges=None):
    availableEnergies = data['energies']
    fixedEnergy = availableEnergies[fixedEnergyIdx]
    energies = np.full(N, fixedEnergy)
    
    if method == 'transformation':
        sampleAngles, sampledEnergies = sampleFromCDF(data=data, material=material, energies=energies, 
                materialToIndex=materialToIndex, cdfs=cdfs, method=method)
    elif method == 'normalization':
        sampleAngles, sampledEnergies = sampleFromCDF(data=data, material=material, energies=energies, 
                materialToIndex=materialToIndex, cdfs=cdfs, binEdges=binEdges, method=method)     
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    dfSamples = pd.DataFrame({
        'angle': sampleAngles,
        'scattered_energy': sampledEnergies
        })
    dfSamples.to_csv(f'SamplingData_{method}_{material}_{fixedEnergy:.1f}MeV.csv', index=False)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    sns.histplot(sampleAngles, bins=100, edgecolor="black", color='orange', kde=False, ax=axs[0])
    axs[0].set_xlabel('Angle')
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Count')
    
    sns.histplot(sampledEnergies, bins=100, color='green', edgecolor='black', kde=False, ax=axs[1])
    axs[1].set_xlabel('Scattered Energy (MeV)')
    axs[1].set_yscale('log')
    axs[1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(f'SamplingDistribution_{method}_{material}_{fixedEnergy:.1f}MeV.pdf')
    plt.close(fig)  
    
# --------------- COMMON FUNCTIONS ---------------
def createSharedMemory(arr : np.ndarray) -> tuple:
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shm_np = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    np.copyto(shm_np, arr)
    return shm, arr.shape, arr.dtype

# --------------- MAIN FUNCTION ----------------
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--transformation', action='store_true', help="Use transformation method")
    group.add_argument('--normalization', action='store_true', help="Use normalization method")
    
    args = parser.parse_args()
    method = 'transformation' if args.transformation else 'normalization'
    
    # Start timing
    startTime = time.time()
    
    # Shared settings
    samplingN = 1000000
    material = 'G4_WATER'
    initialEnergy = 200.  # MeV
    bigVoxelSize = np.array((33.3333, 33.33333, 50), dtype=np.float64)
    voxelShapeBins = (50, 50, 300)
    dt = 1 / 3
    
    angleRange = (0, 70)
    energyRange = (-0.6, 0)
    angleBins = 100
    energyBins = 100

    # Directory setup
    baseFolder = {
        'transformation': {
            'npzPath': './Table/4DTableTrans.npz',
            'savePath': './PlotsSimulation/',
            'npyPath': './Numpy/',
            'csvPath': './CSV/'
        },
        'normalization': {
            'npzPath': './TableNormalized/4DTableNorm.npz',
            'savePath': './PlotsSimulationNormalized/',
            'npyPath': './NumpyNormalized/',
            'csvPath': './CSVNormalized/'
        }
    }

    npzPath = baseFolder[method]['npzPath']
    savePath = baseFolder[method]['savePath']
    npyPath = baseFolder[method]['npyPath']
    csvPath = baseFolder[method]['csvPath']

    # Make sure the timing directory exists
    Path(savePath).mkdir(parents=True, exist_ok=True)
    Path(npyPath).mkdir(parents=True, exist_ok=True)
    Path(csvPath).mkdir(parents=True, exist_ok=True)

    # --- Load table ---
    rawData = np.load(npzPath, mmap_mode='r', allow_pickle=True)
    probTable = rawData['probTable']
    materials = rawData['materials'].tolist()
    energies = rawData['energies']

    # Share data fields
    data = {
        'probTable': probTable,
        'materials': materials,
        'energies': energies
    }
    
    if method == 'normalization':
        # Load extra arrays only used in normalization
        data['thetaMax'] = rawData['thetaMax']
        data['thetaMin'] = rawData['thetaMin']
        data['energyMin'] = rawData['energyMin']
        data['energyMax'] = rawData['energyMax']
        
    rawData.close()
    
    # Build CDFs
    materialToIndex = {mat: idx for idx, mat in enumerate(data['materials'])}      
    if method == 'normalization':
        cdfs, binEdges = buildCdfsAndCompactBins(data=data)
        # Remove unnecessary keys from data to free memory
        for key in ['thetaMax', 'thetaMin', 'energyMin', 'energyMax']:
            del data[key]
        # energy = 15.
        # idx = int(np.argmin(np.abs(energies - energy)))
        # plotSamplingDistribution(data, material, idx, materialToIndex, cdfs, method=method, N=10000000, binEdges=binEdges)
    else:
        cdfs = buildCdfsFromProbTable(probTable)
        # energy = 15.
        # idx = int(np.argmin(np.abs(energies - energy)))
        # plotSamplingDistribution(data, material, idx, materialToIndex, cdfs, method=method, N=10000000)
        
    
    # --- PREBUILD SAMPLERS --- Only for slow interpolation mode
    # prebuildSamplers(
    #     data,
    #     angleRange,
    #     energyRange,
    #     materialToIndex
    # )
    
    # Shared memory
    energyDeposited = np.zeros(voxelShapeBins, dtype=np.float32)
    shm_prob_table, prob_table_shape, prob_table_dtype = createSharedMemory(probTable)
    shm_cdfs, cdfs_shape, cdfs_dtype = createSharedMemory(cdfs)
    shm_energy_deposited, energyDeposited_shape, energyDeposited_dtype = createSharedMemory(energyDeposited)
    
    batchSize = 10000
    numWorkers = cpu_count()
    
    kwargs = {}
    
    if method == 'normalization': 
        # Create shared memories
        shm_bin_edges, bin_edges_shape, bin_edges_type= createSharedMemory(binEdges)
        
        kwargs.update(dict(
            shm_bin_edges = shm_bin_edges,
            bin_edges_dtype = bin_edges_type,
            bin_edges_shape = bin_edges_shape
    ))
        
    elif method == 'transformation':
        # Precompute these values only once, outside of the function
        binningConfig = BinningConfig(angleRange, energyRange, angleBins, energyBins)
        binEdges, angleStep, energyStep = binningConfig.getBinningValues()
        
        kwargs.update(dict(
            binEdges=binEdges,
            angleStep=angleStep,
            energyStep=energyStep
    ))    
        
    # Run simulation
    print(f"Running simulation using '{method}' method.")

    runMultiprocessedBatchedSim(
        samplingN, batchSize, numWorkers,
        shm_prob_table, prob_table_shape, prob_table_dtype,
        shm_cdfs, cdfs_shape, cdfs_dtype,
        shm_energy_deposited, energyDeposited_shape, energyDeposited_dtype,
        data, material, initialEnergy, materialToIndex,
        bigVoxelSize, method, 
        **kwargs
    )
                    
    energyVector3D = np.ndarray(energyDeposited.shape, dtype=energyDeposited.dtype, buffer=shm_energy_deposited.buf).copy()
    np.save(Path(npyPath) / f'projectionXZSimulation_{method}.npy', energyVector3D)
        
    totalEnergy = energyVector3D.sum()
    fileforEnergyDeposit = f"{csvPath}EnergyAtBoxByBinsMySimulation_{method}.csv"
    with open(fileforEnergyDeposit, 'w', newline='') as file:
        # Write the header manually with #
        file.write(
                "# Simulation Version: 4.\n"
                "# Results for scorer: EnergyDeposit\n"
                "# Scored in component: Box\n"
                "# EnergyDeposit (MeV): Sum\n"
                f"# X in {voxelShapeBins[0]} bins of {200 / voxelShapeBins[0]} mm\n"
                f"# Y in {voxelShapeBins[1]} bins of {200 / voxelShapeBins[1]} mm\n"
                f"# Z in {voxelShapeBins[2]} bins of {200 / voxelShapeBins[2]} mm\n"
                f"# Sum : {totalEnergy:.6f} MeV\n"
            )
        writer = csv.writer(file, delimiter=' ')
                    
        # Write voxel data line by line
        for x in range(energyVector3D.shape[0]):
            for y in range(energyVector3D.shape[1]):
                for z in range(energyVector3D.shape[2]):
                    value = energyVector3D[x, y, z]
                    # if value > 0:
                    writer.writerow([x, y, z, f"{value:.6f}"])
                                    
    # Cleanup shared memory     
    shm_prob_table.close()
    shm_prob_table.unlink()
    shm_energy_deposited.close()
    shm_energy_deposited.unlink()
    shm_cdfs.close()
    shm_cdfs.unlink()
    
    if method == 'normalization':
        shm_bin_edges.unlink()
        shm_bin_edges.close()
        
        
    endTime = time.time()
    print(f"Simulation time: {endTime - startTime:.12f} seconds")
    print()