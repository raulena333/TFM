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
import faulthandler
from typing import Tuple
from pathlib import Path

np.set_printoptions(precision=6, suppress=True)

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

# --------------- TRANSFORMATION VARIABLE ---------------       
def sampleFromCDFVectorizedTransformation(data: dict, materials: np.ndarray, energies: np.ndarray, cdfs: np.ndarray,
        binEdges: np.ndarray, angleStep : float, energyStep : float):
    
    # Get the index of the material from the lookup dictionary
    availableEnergies = data['energies']
    
    # Material-specific minimum valid energy (indexed by material index)
    # Index 0 = soft tissue, 1 = water, 2 = bone, 3 = lung
    minEnergyByMaterial = np.array([9.5, 9.0, 12.0, 9.5], dtype=np.float32)
    materialMinEnergies = minEnergyByMaterial[materials]
    
    # Clip energies to the valid range
    minEnergy = np.min(availableEnergies)
    maxEnergy = np.max(availableEnergies)
    roundedEnergies = np.clip(np.round(energies, 1), minEnergy, maxEnergy)
    
    # availableEnergies is descending, so reverse it for searchsorted
    reversedEnergies = availableEnergies[::-1]
    insertPos = np.searchsorted(reversedEnergies, roundedEnergies, side='left')
    insertPos = np.clip(insertPos, 1, len(reversedEnergies) - 1)

    # Compare to both neighbors in reversed array
    left = reversedEnergies[insertPos - 1]
    right = reversedEnergies[insertPos]
    chooseLeft = np.abs(roundedEnergies - left) < np.abs(roundedEnergies - right)

    # Compute correct index back in original (descending) energy array
    closestEnergyIndices = len(availableEnergies) - 1 - np.where(chooseLeft, insertPos - 1, insertPos)
    
    sampledAngles = np.zeros_like(energies, dtype=np.float32)
    sampledEnergies = np.zeros_like(energies, dtype=np.float32)

    # Initialize binning configuration for interpolation of CDF samples
    angleBins, energyBins = data['probTable'].shape[2:]
    
    # Use global binEdges shared across all materials and energies
    angleEdges = binEdges[0, :angleBins + 1]
    energyEdges = binEdges[1, :energyBins + 1]
    
    # Sample in batches for each unique energy index
    uniquePairs = set(zip(materials, closestEnergyIndices))
    for materialIdx, energyIdx in uniquePairs:
        sampleIndices = np.where((materials == materialIdx) & (closestEnergyIndices == energyIdx))[0]
        randValues = np.random.random(size=sampleIndices.size).astype(np.float32)

        sampleCDFForEnergyGroup(
            materialIdx, energyIdx, sampleIndices, randValues,
            cdfs, angleBins, energyBins, angleEdges, energyEdges,
            angleStep, energyStep, sampledAngles, sampledEnergies
        )

    # Apply transformation back to physical space
    realAngles, realEnergies = reverseVariableChangeTransform(energies, sampledAngles, sampledEnergies)
    
    # For input energies below the minimum threshold, zero out the output
    mask = energies < materialMinEnergies
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

# --------------- NORMALIZATION VARIABLE -------------
@numba.jit(nopython=True, inline = 'always') 
def catMullRomInterpolation(f_1, f0, f1, f2, t):    
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * f0) + 
        (-f_1 + f1) * t + 
        (2.0 * f_1 - 5.0 * f0 + 4.0 * f1 - f2) * t2 + 
        (-f_1 + 3.0 * f0 - 3.0 * f1 +f2) * t3
    ) 
    
# --------------- NORMALIZATION VARIABLE -------------
def sampleFromCDFVectorizedNormalizationCubic(
    data: dict, 
    materials: np.ndarray, 
    energies: np.ndarray, 
    cdfs: np.ndarray, 
    binEdges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    
    availableEnergies = data['energies']
    angleBins, energyBins = data['probTable'].shape[2:]
    
    # Material-specific minimum valid energy (indexed by material index)
    # Index 0 = soft tissue, 1 = water, 2 = bone, 3 = lung
    minEnergyByMaterial = np.array([9.5, 9.0, 12.0, 9.5], dtype=np.float32)
    materialMinEnergies = minEnergyByMaterial[materials]

    minEnergy = np.min(availableEnergies)
    maxEnergy = np.max(availableEnergies)
    roundedEnergies = np.clip(energies, minEnergy, maxEnergy)

    # Interpolation setup
    reversedEnergies = availableEnergies[::-1]
    insertPos = np.searchsorted(reversedEnergies, roundedEnergies, side='left')
    insertPos = np.clip(insertPos, 1, len(reversedEnergies) - 2)

    base = len(availableEnergies) - 1
    i1 = base - insertPos
    i0 = np.clip(i1 - 1, 0, base)
    i2 = np.clip(i1 + 1, 0, base)
    i_1 = np.clip(i1 - 2, 0, base)

    e0 = availableEnergies[i0]
    e1 = availableEnergies[i1]

    # Interpolation weights, only applied where energy >= minEnergy
    weights = np.where(
        e1 != e0,
        (roundedEnergies - e0) / (e1 - e0),
        0.0
    ).astype(np.float32)

    rand = np.random.random(size=energies.shape).astype(np.float32)

    out = {}
    for label, indices in zip(
        ['_1', '0', '1', '2'], [i_1, i0, i1, i2]
    ):
        out[label + '_angles'] = np.zeros_like(energies, dtype=np.float32)
        out[label + '_energies'] = np.zeros_like(energies, dtype=np.float32)

        uniquePairs = set(zip(materials, indices))
        for materialIdx, energyIdx in uniquePairs:
            idxs = np.where((materials == materialIdx) & (indices == energyIdx))[0].astype(np.int32)
            
            angleEdges = binEdges[materialIdx, energyIdx, 0, :angleBins + 1]
            energyEdges = binEdges[materialIdx, energyIdx, 1, :energyBins + 1]

            if angleEdges[0] == angleEdges[-1] or energyEdges[0] == energyEdges[-1]:
                out[label + '_angles'][idxs] = 0.0
                out[label + '_energies'][idxs] = 0.0
                continue

            angleStep = angleEdges[1] - angleEdges[0]
            energyStep = energyEdges[1] - energyEdges[0]

            sampleCDFForEnergyGroup(
                materialIdx, energyIdx, idxs, rand[idxs],
                cdfs, angleBins, energyBins, angleEdges, energyEdges,
                angleStep, energyStep, out[label + '_angles'], out[label + '_energies']
            )

    interpolateMask = energies >= materialMinEnergies
    sampledAngles = np.where(
        interpolateMask,
        catMullRomInterpolation(
            out['_1_angles'], out['0_angles'], out['1_angles'], out['2_angles'], weights
        ),
        0.0
    )
    sampledEnergies = np.where(
        interpolateMask,
        catMullRomInterpolation(
            out['_1_energies'], out['0_energies'], out['1_energies'], out['2_energies'], weights
        ),
        0.0
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
def getVoxelIndex(position, size, bins):
    
    ix = int((position[0] + size[0]) * bins[0] / (2 * size[0]))
    iy = int((position[1] + size[1]) * bins[1] / (2 * size[1]))
    iz = int((position[2] + size[2]) * bins[2] / (2 * size[2]))

    ix = min(max(ix, 0), bins[0] - 1)
    iy = min(max(iy, 0), bins[1] - 1)
    iz = min(max(iz, 0), bins[2] - 1)
    # print(f"Voxel Index: ({ix}, {iy}, {iz})")

    return ix, iy, iz

# --------------- COMMON FUNCTIONS ---------------
@numba.jit(nopython=True)
def getMaterialIndexAtPosition(positions, grid, physicalSize, voxelShapeBins):
    n = positions.shape[0]
    indices = np.empty(n, dtype=np.int32)
    
    for i in range(n):
        ix, iy, iz = getVoxelIndex(positions[i], physicalSize, voxelShapeBins)
        indices[i] = grid[ix, iy, iz]
        
    return indices

# --------------- COMMON FUNCTIONS ---------------
def sampleFromCDF(
    data, materialsIndex, energies, cdfs, binEdges, method="transformation",
    angleStep=None, energyStep=None
):
    if method == "transformation":
        return sampleFromCDFVectorizedTransformation(
            data=data,
            materials=materialsIndex,
            energies=energies,
            cdfs=cdfs,
            binEdges=binEdges,
            angleStep=angleStep,
            energyStep=energyStep
        )
    elif method == "normalization":
        if binEdges is None:
            raise ValueError("Bin edges must be provided for 'normalization' method.")
        return sampleFromCDFVectorizedNormalizationCubic(
            data=data,
            materials=materialsIndex,
            energies=energies,
            cdfs=cdfs,
            binEdges=binEdges
        )
    else:
        raise ValueError(f"Unknown sampling method: {method}")


# --------------- COMMON FUNCTIONS ---------------   
def sampling3dStepTraversal(voxelIndices, segmentLengths, segmentCounts, grid,
    data, energies, cdfs, binEdges, method, angleStep, energyStep
    ):
    n, _, _ = voxelIndices.shape
    
    samples = np.zeros(n, dtype=np.float64)
    energy = np.zeros(n, dtype=np.float64)

    unique_counts = np.unique(segmentCounts)
    
    for count in unique_counts:
        if count == 0:
            # particles with zero segments get zero result
            samples[segmentCounts == 0] = 0.0
            energy[segmentCounts == 0] = 0.0
            continue
        
        # Mask particles with this count:
        mask = (segmentCounts == count)
        indices = np.where(mask)[0]

        # Extract their relevant segments:
        seg_voxels = voxelIndices[indices, :count, :]  # shape (m, count, 3)
        seg_lengths = segmentLengths[indices, :count]  # shape (m, count)

        # Flatten for efficient material lookup:
        flat_voxels = seg_voxels.reshape(-1, 3)

        # Get materials for all those voxels:
        materials = grid[flat_voxels[:, 0], flat_voxels[:, 1], flat_voxels[:, 2]]

        repeatedEnergies = np.repeat(energies[indices], count)
        theta_samples, energy_samples = sampleFromCDF(data=data, materialsIndex=materials, energies=repeatedEnergies,
            cdfs=cdfs, binEdges=binEdges, method=method, 
            angleStep=angleStep, energyStep=energyStep
        )
        # print(f"Sampling for count {count}: theta_samples {theta_samples}, energy_samples {energy_samples}")

        # Reshape back to (m, count)
        theta_samples = theta_samples.reshape(len(indices), count)
        energy_samples = energy_samples.reshape(len(indices), count)

        # Weighted sums:
        weighted_theta = np.sum(theta_samples * seg_lengths, axis=1)
        weighted_energy = np.sum(energy_samples * seg_lengths, axis=1)
        total_len = np.sum(seg_lengths, axis=1)

        # Weighted averages:
        samples[indices] = weighted_theta / total_len
        energy[indices] = weighted_energy / total_len

    return samples, energy

# --------------- COMMON FUNCTIONS ---------------
@numba.jit(nopython=True, fastmath=True)
def dda3dStepTraversal(initialPositions,
    directions,
    stepLength,
    grid,
    gridShape,
    physicalSize,
    max_segments=5
    ):
    n = initialPositions.shape[0]
    binsX, binsY, binsZ = gridShape
    sizeX, sizeY, sizeZ = physicalSize
    
    # voxelSize = np.array([2 * sizeX / binsX, 2 * sizeY / binsY, 2 * sizeZ / binsZ])
    # gridOrigin = np.array([-sizeX, -sizeY, -sizeZ])
    voxelSize = np.empty(3, dtype=np.float64)
    voxelSize[0] = 2 * sizeX / binsX
    voxelSize[1] = 2 * sizeY / binsY
    voxelSize[2] = 2 * sizeZ / binsZ

    gridOrigin = np.empty(3, dtype=np.float64)
    gridOrigin[0] = -sizeX
    gridOrigin[1] = -sizeY
    gridOrigin[2] = -sizeZ

    voxelIndices = -np.ones((n, max_segments, 3), dtype=np.int32)
    segmentLengths = np.zeros((n, max_segments), dtype=np.float32)
    segmentCounts = np.zeros(n, dtype=np.int32)

    for i in range(n):
        pos = initialPositions[i]
        dir = directions[i]
        traveled = 0.0

        # Get initial voxel index:
        # voxel = np.array(getVoxelIndex(pos, physicalSize, gridShape))
        voxel_idx = getVoxelIndex(pos, physicalSize, gridShape)
        voxel = np.empty(3, dtype=np.int32)
        voxel[0], voxel[1], voxel[2] = voxel_idx[0], voxel_idx[1], voxel_idx[2]
        
        if np.any(voxel < 0) or voxel[0] >= binsX or voxel[1] >= binsY or voxel[2] >= binsZ:
            # Out of bounds initially, skip or handle differently
            continue
        
        # Compute step, tMax, tDelta per axis:
        step = np.empty(3, dtype=np.int32)
        t_max = np.empty(3, dtype=np.float64)
        t_delta = np.empty(3, dtype=np.float64)

        for j in range(3):
            if abs(dir[j]) < 1e-20:
                step[j] = 0
                t_max[j] = 1e30
                t_delta[j] = 1e30
            else:
                if dir[j] > 0:
                    step[j] = 1
                    next_boundary = gridOrigin[j] + (voxel[j] + 1) * voxelSize[j]
                else:
                    step[j] = -1
                    next_boundary = gridOrigin[j] + voxel[j] * voxelSize[j]
                t_max[j] = (next_boundary - pos[j]) / dir[j]
                t_delta[j] = voxelSize[j] / abs(dir[j])

        count = 0

        while traveled < stepLength and count < max_segments:
            # Current voxel is traversed for at least:
            min_t_max = t_max[0]
            axis = 0
            if t_max[1] < min_t_max:
                min_t_max = t_max[1]
                axis = 1
            if t_max[2] < min_t_max:
                min_t_max = t_max[2]
                axis = 2
            
            # Distance to next voxel boundary or remaining step length:
            travel_len = min(min_t_max - traveled, stepLength - traveled)

            # Store voxel and segment info if inside grid:
            ix, iy, iz = voxel
            if 0 <= ix < binsX and 0 <= iy < binsY and 0 <= iz < binsZ:
                material = grid[ix, iy, iz]
                voxelIndices[i, count] = voxel
                segmentLengths[i, count] = travel_len
                # print(f'Travel new segment:', count, 'at voxel:', ix, iy, iz, 'length:', travel_len, 'material:', material)
                count += 1
            else:
                # Out of bounds voxel - stop traversal for this particle
                break

            traveled += travel_len
            if traveled >= stepLength:
                break

            # Move to next voxel along axis:
            voxel[axis] += step[axis]
            t_max[axis] += t_delta[axis]

        segmentCounts[i] = count

    return voxelIndices, segmentLengths, segmentCounts

# --------------- COMMON FUNCTIONS ---------------
@numba.jit(nopython=True, fastmath=True)
def depositEnergy3DStepTraversal(
    voxelIndices, segmentLengths, segmentCounts, energyLossPerStep, energyDepositedVector
):
    for i in range(len(segmentCounts)):
        count = segmentCounts[i]
        if count == 0:
            continue

        total_len = 0.0
        for j in range(count):
            total_len += segmentLengths[i, j]
        if total_len <= 0.0:
            continue

        for j in range(count):
            frac = segmentLengths[i, j] / total_len
            deposit = frac * energyLossPerStep[i]
            x, y, z = voxelIndices[i, j]
            energyDepositedVector[x, y, z] += deposit
            # print(f'Depositing energy: {deposit} at voxel ({x}, {y}, {z})')
        
# --------------- COMMON FUNCTIONS ---------------
def simulateBatchParticlesVectorized(
    batchSize, data, gridMap, initialEnergy,
    bigVoxelSize, energyDepositedVector, cdfs, 
    binEdges, method='transformation',
    angleStep=None, energyStep=None
):
    initialPosition = -150.0  # Initial position in mm
    energy = np.full(batchSize, initialEnergy)
    position = np.tile([0.0, 0.0, initialPosition], (batchSize, 1))
    velocity = np.tile([0.0, 0.0, 1.0], (batchSize, 1))
    active = np.ones(batchSize, dtype=bool)
    # print(f'Initial Energy: {initialEnergy} MeV')

    zStep = 1  # Fixed step of 1 mm
    while np.any(active):
        energyActive = energy[active]
        
        # materialsIndex = getMaterialIndexAtPosition(
        #     positions=position[active], grid=gridMap, physicalSize=bigVoxelSize,
        #     voxelShapeBins=gridMap.shape) 
        
        # Get voxel indices and segment lengths using DDA traversal
        voxelIndices, segmentLengths, segmentCounts = dda3dStepTraversal(
            initialPositions=position[active], directions=velocity[active],
            stepLength=zStep, grid=gridMap, gridShape=gridMap.shape,
            physicalSize=bigVoxelSize, max_segments=5
        )
        
        # print(f'Voxel Indices: {voxelIndices}')
        # print(f'Segment Lengths: {segmentLengths}')
        # print(f'Segment Counts: {segmentCounts}')
        
    
        # Use weighted sampling based on segment lengths
        realAngles, realEnergies = sampling3dStepTraversal(
            voxelIndices=voxelIndices, segmentLengths=segmentLengths,
            segmentCounts=segmentCounts, grid=gridMap, data=data, energies=energyActive,
            cdfs=cdfs, binEdges=binEdges, method=method,
            angleStep=angleStep, energyStep=energyStep
        )
        energyLossPerStep = energyActive - realEnergies
        energy[active] = realEnergies
        
        # print(f'Energy Active: {energy[active]} MeV')
        # print(f'Position: {position[active]} mm')
        # print(f'Direction: {velocity[active]}')
        
        # print(f'Sampled Energies: {realEnergies} MeV')
        # print(f'Sampled Theta Angles: {realAngles} degrees')
        # print(f'Energy Loss per Step: {energyLossPerStep} MeV')
    
        # Deposit energy in the energyDepositedVector
        depositEnergy3DStepTraversal(
            voxelIndices=voxelIndices, segmentLengths=segmentLengths,
            segmentCounts=segmentCounts, energyLossPerStep=energyLossPerStep,
            energyDepositedVector=energyDepositedVector)

        position[active], velocity[active] = updatePosition(
            position[active], velocity[active], zStep, realAngles
        )
                
        # Check if particles are still active
        withinBounds = np.all(
            (position[active] >= -bigVoxelSize) & (position[active] <= bigVoxelSize),
            axis=1
        )
        realEnergiesValid = realEnergies > 0
        active[active] = realEnergiesValid & withinBounds

       
# --------------- COMMON FUNCTIONS ---------------
def updatePosition(position, velocity, step_length, realAngles):
    n = position.shape[0]

    # Sample phi azimuth angles uniformly
    phi = np.random.uniform(0, 2 * np.pi, n)
    theta = np.radians(realAngles)  # convert degrees to radians

    v = velocity  # old velocities, shape (n,3)

    zAxis = np.array([0.0, 0.0, 1.0])
    perp = np.cross(v, zAxis)
    norm_perp = np.linalg.norm(perp, axis=1, keepdims=True)
    small_mask = (norm_perp[:, 0] < 1e-8)
    perp[small_mask] = np.array([1.0, 0.0, 0.0])
    norm_perp[small_mask] = 1.0
    perp /= norm_perp

    cos_phi = np.cos(phi)[:, None]
    sin_phi = np.sin(phi)[:, None]
    k = perp * cos_phi + np.cross(v, perp) * sin_phi
    k /= np.linalg.norm(k, axis=1, keepdims=True)  # normalize rotation axis

    cos_theta = np.cos(theta)[:, None]
    sin_theta = np.sin(theta)[:, None]
    # Rodrigues' rotation formula
    v_rot = (v * cos_theta +
             np.cross(k, v) * sin_theta +
             k * (np.sum(k * v, axis=1, keepdims=True)) * (1 - cos_theta))
    v_rot /= np.linalg.norm(v_rot, axis=1, keepdims=True)  # normalize direction

    # Move particle exactly step_length along rotated direction
    displacement = v_rot * step_length
    new_position = position + displacement
    new_velocity = v_rot

    return new_position, new_velocity
        
# --------------- COMMON FUNCTIONS ---------------
def simulateBatchParticlesWorker(args):
    method = args[16] 
    
    # Common extraction for all methods
    (
        shm_name, shape, dtype_str,
        shm_cdfs_name, cdfs_shape, cdfs_dtype_str,
        shm_energy_deposited_name, energy_deposited_shape, energy_deposited_dtype_str,
        batchSize, materials, energies,
        gridMaterial, initialEnergy,
        bigVoxelSize, seed, _  # method is already extracted
    ) = args[:17]

    # Default to None (for both methods)
    binEdges = angleStep = energyStep = None
    shm_bin_edges_name = bin_edges_shape = bin_edges_dtype_str = None

    # Handle method-specific arguments
    if method == 'normalization':
        (
            shm_bin_edges_name, bin_edges_shape, bin_edges_dtype_str
        ) = args[17:]

    elif method == 'transformation':
        (
            binEdges, angleStep, energyStep
        ) = args[17:]

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
        
    return simulateBatchParticlesVectorized(batchSize=batchSize, data=data, gridMap=gridMaterial, initialEnergy=initialEnergy,
            bigVoxelSize=bigVoxelSize, energyDepositedVector=energyDepositedVector, cdfs=cdfs,
            binEdges=binEdges, method=method, angleStep=angleStep, energyStep=energyStep)

# --------------- COMMON FUNCTIONS ---------------
def runMultiprocessedBatchedSim(
    totalSamples, batchSize, numWorkers,
    shm_prob_table, prob_table_shape, prob_table_dtype,
    shm_cdfs, cdfs_shape, cdfs_dtype,
    shm_energy_deposited, energy_deposited_shape, energy_deposited_dtype,
    data, gridMaterial, initialEnergy, 
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
    
    baseSeed = 767435635
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
            gridMaterial, initialEnergy,
            bigVoxelSize, currentSeed, # seed for each batch, reproducible
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
    initialEnergy = 200.  # MeV
    bigVoxelSize = np.array((100., 100., 150.), dtype=np.float32)
    voxelShapeBins = (50, 50, 300)
    
    angleRange = (0, 70)
    energyRange = (-0.6, 0)
    angleBins = 100
    energyBins = 100
    
    # Load grid of heterogeneous materials
    gridMaterial = np.zeros((voxelShapeBins[0], voxelShapeBins[1], voxelShapeBins[2]), dtype=np.int32)
    gridPath = '../TableMaterials/materialGrid.npy'
    if not os.path.exists(gridPath):
        raise FileNotFoundError(f"Grid file not found at {gridPath}. Please ensure the grid is generated first. Loading default grid with zeros.")
    else:
        print(f"Loading grid from {gridPath}")
        gridMaterial = np.load(gridPath, allow_pickle=True)
    
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
    if method == 'normalization':
        cdfs, binEdges = buildCdfsAndCompactBins(data=data)
        # Remove unnecessary keys from data to free memory
        for key in ['thetaMax', 'thetaMin', 'energyMin', 'energyMax']:
            del data[key]
    else:
        cdfs = buildCdfsFromProbTable(probTable)

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
        data, gridMaterial, initialEnergy,
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