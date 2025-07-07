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
import seaborn as sns

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

# --------------- TRANSFORMATION VARIABLE --------------
def sampleFromCDFVectorizedTransformation(data: dict, materials: np.ndarray, energies: np.ndarray, cdfs: np.ndarray,
        binEdges: np.ndarray, angleStep : float, energyStep : float):
    
    # Get the index of the material from the lookup dictionary
    availableEnergies = data['energies']
    escapeProbs = data['escapeProbs'] 
    
    # Material-specific minimum valid energy (indexed by material index)
    # Index 0 = lung, 1 = water, 2 = bone, 3 = soft tissue
    minEnergyByMaterial = np.array([8.9, 8.7, 11.6, 8.9], dtype=np.float32)
    maxEnergyByMaterial = np.array([9.2, 9.0, 12.0, 9.2], dtype=np.float32)
    
    # Initialize output
    sampledAngles = np.zeros_like(energies, dtype=np.float32)
    sampledEnergies = np.zeros_like(energies, dtype=np.float32)
    closestEnergyIndices = np.full_like(materials, fill_value=-1, dtype=np.int16)

    # Initialize binning configuration for interpolation of CDF samples
    angleBins, energyBins = data['probTable'].shape[2:]
    
    # Use global binEdges shared across all materials and energies
    angleEdges = binEdges[0, :angleBins + 1]
    energyEdges = binEdges[1, :energyBins + 1]
    
    # Determine survival mask
    survived = np.ones_like(energies, dtype=bool)
    materialMinEnergies = minEnergyByMaterial[materials]
    materialMaxEnergies = maxEnergyByMaterial[materials]
    
    # Below min energy always die
    belowMinMasK = energies < materialMinEnergies
    survived[belowMinMasK] = False
    # Above max energy always survive
    aboveMaxMask = energies > materialMaxEnergies
    # In range mask depends
    inRangeMask = (~belowMinMasK) & (~aboveMaxMask)
    #print('In Range Mask', inRangeMask)
    
    # Clip energies to the valid range
    minEnergy = np.min(availableEnergies)
    maxEnergy = np.max(availableEnergies)
    roundedEnergies = np.clip(np.round(energies, 1), minEnergy, maxEnergy)
    
    inRangeIndices = np.where(inRangeMask)[0]
    
    # Generate random numbers for these particles
    randVals = np.random.random(size=inRangeIndices.size)
    if np.any(inRangeMask):
        uniqueInRange = set(zip(materials[inRangeIndices], roundedEnergies[inRangeIndices]))
        for material, energy in uniqueInRange:
            maskUnique = (materials[inRangeIndices] == material) & (roundedEnergies[inRangeIndices] == energy)
            particleIndices = inRangeIndices[maskUnique]
            
            # Get escape table for this material
            escapeTable = escapeProbs[material]
            eTable = escapeTable[:, 0]
            pTable = escapeTable[:, 1]
            
            # Find closest index in eTable for this energy
            closestIdx = np.argmin(np.abs(eTable - energy))
            #print('closestIdx', closestIdx, eTable[closestIdx])
            survivalProb = pTable[closestIdx]
            #print('survivalProb', survivalProb)
            
            survived[particleIndices] = randVals[maskUnique] < survivalProb
    
    sampledMask = survived     
    if np.any(sampledMask):  
        # Use roundedEnergies filtered to sampleMask only
        filteredRoundedEnergies = roundedEnergies[sampledMask]
        reversedEnergies = availableEnergies[::-1]
        
        insertPos = np.searchsorted(reversedEnergies, filteredRoundedEnergies, side='left')
        insertPos = np.clip(insertPos, 1, len(reversedEnergies) - 1)
        
        left = reversedEnergies[insertPos - 1]
        right = reversedEnergies[insertPos]
        chooseLeft = np.abs(filteredRoundedEnergies - left) < np.abs(filteredRoundedEnergies - right)
        closestEnergyIndices[sampledMask] = len(availableEnergies) - 1 - np.where(chooseLeft, insertPos - 1, insertPos)
        #print('closestEnergyIndices', closestEnergyIndices[sampleMask])
        
        validMaterials = materials[sampledMask]
        validEnergiesIndices = closestEnergyIndices[sampledMask]
        
        uniquePairs = set(zip(validMaterials, validEnergiesIndices))
        globalIndices = np.flatnonzero(sampledMask)  # global indices of all samples passing mask
        for materialIdx, energyIdx in uniquePairs:
            #print('materialIdx', materialIdx, 'energyIdx', energyIdx)
            sampleIndices = np.where((validMaterials == materialIdx) & (validEnergiesIndices == energyIdx))[0]
            #print('Sample indices', sampleIndices)
            randValues = np.random.random(size=sampleIndices.size).astype(np.float32)
            
            # Convert to global indices into full arrays
            globalSampleIndices = globalIndices[sampleIndices]
            
            randValues = np.random.random(size=sampleIndices.size).astype(np.float32)
            
            sampleCDFForEnergyGroup(
                materialIdx, energyIdx, globalSampleIndices, randValues,
                cdfs, angleBins, energyBins, angleEdges, energyEdges,
                angleStep, energyStep, sampledAngles, sampledEnergies
            )
            
    sampledAngles[~survived] = 0
    sampledEnergies[~survived] = 0
             
    realAngles, realEnergies = reverseVariableChangeTransform(energies, sampledAngles, sampledEnergies)
    
    # print(f'Real energies: {realEnergies}')

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
def sampleFromCDFVectorizedNormalizationCubic(
    data: dict, 
    materials: np.ndarray, 
    energies: np.ndarray, 
    cdfs: np.ndarray, 
    binEdges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    
    availableEnergies = data['energies']
    escapeProbs = data['escapeProbs']
    angleBins, energyBins = data['probTable'].shape[2:]
    
    # Material-specific minimum valid energy (indexed by material index)
    # Index 0 = lung , 1 = water, 2 = bone, 3 = soft tissue
    minEnergyByMaterial = np.array([8.9, 8.7, 11.6, 8.9], dtype=np.float32)
    maxEnergyByMaterial = np.array([9.2, 9.0, 12.0, 9.2], dtype=np.float32)
    
    # Initialize output
    sampledAngles = np.zeros_like(energies, dtype=np.float32)
    sampledEnergies = np.zeros_like(energies, dtype=np.float32)
    
    # Survival mask
    survived = np.ones_like(energies, dtype=bool)
    sampledAngles = np.zeros_like(energies, dtype=np.float32)
    sampledEnergies = np.zeros_like(energies, dtype=np.float32)
    
    materialMinEnergies = minEnergyByMaterial[materials]
    materialMaxEnergies = maxEnergyByMaterial[materials]
    
    # Below min energy always die
    belowMinMasK = energies < materialMinEnergies
    survived[belowMinMasK] = False
    # Above max energy always survive
    aboveMaxMask = energies > materialMaxEnergies
    # In range mask depends
    inRangeMask = (~belowMinMasK) & (~aboveMaxMask)
    #print('In Range Mask', inRangeMask)

    minEnergy = np.min(availableEnergies)
    maxEnergy = np.max(availableEnergies)
    roundedEnergies = np.clip(energies, minEnergy, maxEnergy)
    
    inRangeIndices = np.where(inRangeMask)[0]
    # Generate random numbers for these particles
    randVals = np.random.random(size=inRangeIndices.size)
    
    if np.any(inRangeMask):
        # Get unique materials present in the in-range particles
        uniqueMaterials = np.unique(materials[inRangeIndices])
        for material in uniqueMaterials:
            materialMask = (materials[inRangeIndices] == material)
            particleIndices = inRangeIndices[materialMask]
            
            # Get escape table for this material
            escapeTable = escapeProbs[material]
            eTable = escapeTable[:, 0]
            pTable = escapeTable[:, 1]
            
            # Energies of the particles
            particleEnergies = roundedEnergies[particleIndices]
            
            # Initializate array
            survivalProbs = np.zeros_like(particleEnergies, dtype=np.float32)
            # For each particle energy, find index i and weight t for Catmull-Rom
            for idx, energyVal in enumerate(particleEnergies):
                i = np.searchsorted(eTable, energyVal, side='left')
                i = max(1, min(i, len(eTable) - 3))
                t = (energyVal - eTable[i]) / (eTable[i + 1] - eTable[i])

                f_1, f0, f1, f2 = getCatmullRomPoints(pTable, i)
                survivalProbs[idx] = catMullRomInterpolation(f_1, f0, f1, f2, t)
                
            # generate survival mask
            survived[particleIndices] = randVals[materialMask] < survivalProbs
    
    sampledMask = survived
    
    if np.any(sampledMask):
        # Interpolation setup
        filteredRoundedEnergies = roundedEnergies[sampledMask]
        filteredMaterials = materials[sampledMask]
        
        reversedEnergies = availableEnergies[::-1]
        insertPos = np.searchsorted(reversedEnergies, filteredRoundedEnergies, side='left')
        insertPos = np.clip(insertPos, 1, len(reversedEnergies) - 2)

        base = len(availableEnergies) - 1
        i1 = base - insertPos
        i0 = np.clip(i1 - 1, 0, base)
        i2 = np.clip(i1 + 1, 0, base)
        i_1 = np.clip(i1 - 2, 0, base)
        # print('Initial energy to interpolate:', filteredRoundedEnergies)
        # print('Energies to interpolate:')
        # print('i0:', availableEnergies[i0], 'i1:', availableEnergies[i1], 'i2:', availableEnergies[i2])
        
        e0 = availableEnergies[i0]
        e1 = availableEnergies[i1]

        # Interpolation weights, only applied where energy >= minEnergy
        weights = (filteredRoundedEnergies - e0) / (e1 - e0)
        rand = np.random.random(size=filteredRoundedEnergies.shape).astype(np.float32)

        out = {}
        for label, indices in zip(
            ['_1', '0', '1', '2'], [i_1, i0, i1, i2]
        ):
            out[label + '_angles'] = np.zeros_like(filteredRoundedEnergies, dtype=np.float32)
            out[label + '_energies'] = np.zeros_like(filteredRoundedEnergies, dtype=np.float32)

            uniquePairs = set(zip(filteredMaterials, indices))
            for materialIdx, energyIdx in uniquePairs:
                idxs = np.where((filteredMaterials == materialIdx) & (indices == energyIdx))[0].astype(np.int32)
                
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

        interpolatedAngles = catMullRomInterpolation(
            out['_1_angles'], out['0_angles'], out['1_angles'], out['2_angles'], weights
        )
        interpolatedEnergies = catMullRomInterpolation(
            out['_1_energies'], out['0_energies'], out['1_energies'], out['2_energies'], weights
        )

        # Assign only to the sampledMask positions
        sampledAngles[sampledMask] = interpolatedAngles
        sampledEnergies[sampledMask] = interpolatedEnergies
        
    sampledAngles[~survived] = 0
    sampledEnergies[~survived] = 0

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
@numba.jit(nopython=True, fastmath=True)
def dda3dStepTraversal(initialPositions,
    directions,
    grid,
    gridShape,
    physicalSize,
    materialIndices,
    segmentLengths,
    segmentCounts,
    voxelIndices,
    stepLength=1.0,
    maxSegments=5
    ):
    n = initialPositions.shape[0]
    binsX, binsY, binsZ = gridShape
    sizeX, sizeY, sizeZ = physicalSize
    
    voxelSize = np.empty(3, dtype=np.float64)
    voxelSize[0] = 2 * sizeX / binsX
    voxelSize[1] = 2 * sizeY / binsY
    voxelSize[2] = 2 * sizeZ / binsZ

    gridOrigin = np.empty(3, dtype=np.float64)
    gridOrigin[0] = -sizeX
    gridOrigin[1] = -sizeY
    gridOrigin[2] = -sizeZ

    for i in range(n):
        pos = initialPositions[i]
        dir = directions[i]
        pos = pos + 1e-6 * dir
        traveled = 0.0

        # Get initial voxel index:
        voxelIndx = getVoxelIndex(pos, physicalSize, gridShape)
        print(f'Voxel index for particle {i}:', voxelIndx)
        voxel = np.empty(3, dtype=np.int32)
        voxel[0], voxel[1], voxel[2] = voxelIndx[0], voxelIndx[1], voxelIndx[2]
        
        if np.any(voxel < 0) or voxel[0] >= binsX or voxel[1] >= binsY or voxel[2] >= binsZ:
            # Out of bounds initially, skip or handle differently
            continue
        
        # Compute step, tMax, tDelta per axis:
        step = np.empty(3, dtype=np.int32)
        tMax = np.empty(3, dtype=np.float64)
        tDelta = np.empty(3, dtype=np.float64)

        for j in range(3):
            if abs(dir[j]) < 1e-20:
                step[j] = 0
                tMax[j] = np.inf
                tDelta[j] = np.inf
            else:
                if dir[j] > 0:
                    step[j] = 1
                    nextBoundary = gridOrigin[j] + (voxel[j] + 1) * voxelSize[j]
                else:
                    step[j] = -1
                    nextBoundary = gridOrigin[j] + voxel[j] * voxelSize[j]
                tMax[j] = (nextBoundary - pos[j]) / dir[j]
                tDelta[j] = voxelSize[j] / abs(dir[j])

        count = 0

        while traveled < stepLength and count < maxSegments:
            # Current voxel is traversed for at least:
            mintMax = tMax[0]
            axis = 0
            if tMax[1] < mintMax:
                mintMax = tMax[1]
                axis = 1
            if tMax[2] < mintMax:
                mintMax = tMax[2] 
                axis = 2
            
            # Distance to next voxel boundary or remaining step length:
            travelLength = min(mintMax, stepLength - traveled)
            
            # Fix floating point inaccuracies (e.g., 0.999999 -> 1.0)
            if np.isclose(travelLength, stepLength - traveled, atol=1e-5):
                travelLength = stepLength - traveled

            # Store voxel and segment info if inside grid:
            ix, iy, iz = voxel
            if 0 <= ix < binsX and 0 <= iy < binsY and 0 <= iz < binsZ:
                material = grid[ix, iy, iz]
                materialIndices[i, count] = material
                segmentLengths[i, count] = travelLength
                voxelIndices[i, count, :] = voxel
                # print(f'Travel new segment:', count, 'at voxel:', ix, iy, iz, 'length:', travelLength, 'material:', material)
                count += 1
                
                traveled += travelLength

                # Move to next voxel along axis:
                voxel[axis] += step[axis]
                tMax[axis] += tDelta[axis]
            else:
                # Out of bounds voxel - stop traversal for this particle
                traveled += travelLength
                break

        segmentCounts[i] = count

    return materialIndices, segmentLengths, segmentCounts
        
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

def scatteringStepMaterial(
    initialPositions,
    directions,
    initialEnergies,
    energyGrid,
    fluenceGrid,
    energyFluenceGrid,
    grid,
    data,
    cdfs,
    binEdges,
    gridShape,
    physicalSize,
    method,
    angleStep,
    energyStep,
    stepLength=1.0,
    maxSegments=5
):
    numberOfParticles = initialPositions.shape[0]
    
    newPositions = initialPositions.copy()
    newDirections = directions.copy()
    newEnergies = initialEnergies.copy()
    
    # Preallocate output arrays
    materialIndices = -np.ones((numberOfParticles, maxSegments), dtype=np.int32)
    segmentLengths = np.zeros((numberOfParticles, maxSegments), dtype=np.float32)
    segmentCounts = np.zeros(numberOfParticles, dtype=np.int32)
    voxelIndices = -np.ones((numberOfParticles, maxSegments, 3), dtype=np.int32)
    
    # First we sample angles and energies
    materialIndex = getMaterialIndexAtPosition(
        positions=initialPositions,
        grid=grid,
        physicalSize=physicalSize,
        voxelShapeBins=gridShape
    )
    # print('Material indices:', materialIndex)
    
    # Sample angles and energies
    sampleAngles, sampledEnergies = sampleFromCDF(
        data=data,
        materialsIndex=materialIndex,
        energies=initialEnergies,
        cdfs=cdfs,
        binEdges=binEdges,
        method=method,
        angleStep=angleStep,
        energyStep=energyStep
    )
    # print('Sampled angles:', sampleAngles)
    # print('Sampled energies:', sampledEnergies)
    
    #Update only directions based on angles sampled
    newDirections = updateDirection(
        velocity=directions,
        realAngles=sampleAngles
    )
    #print('New directions:\n', newDirections)

    # Get material indices, segment lengths, segment counts, and voxel indices
    # for each particle we take the initial direction and position
    dda3dStepTraversal(
        initialPositions=initialPositions,
        directions=directions,
        grid=grid,
        gridShape=gridShape,
        physicalSize=physicalSize,
        materialIndices=materialIndices,
        segmentLengths=segmentLengths,
        segmentCounts=segmentCounts,
        voxelIndices=voxelIndices,
        stepLength=stepLength,
        maxSegments=maxSegments
    )
    # print('Material indices:', materialIndices)
    # print('Segment lengths:', segmentLengths)
    # print('Segment counts:', segmentCounts)
    # print('Voxel indices:', voxelIndices)
    
    # Compute valid mask and first material per particle
    validMask = np.arange(maxSegments) < segmentCounts[:, None]
    firstMaterial = materialIndices[:, 0][:, None]

    # Determine which particles stayed in a single material
    sameMaterialMask = (materialIndices == firstMaterial) | ~validMask
    allSame = np.all(sameMaterialMask, axis=1)

    crossedMultipleMaterials = ~allSame
    stayedSingleMaterial = allSame

    # Divide particles into those that stayed in a single material and those that crossed multiple
    # If stayed in single material, the stepLength is used completely = 1. 
    if np.any(stayedSingleMaterial):
        singleIndices = np.where(stayedSingleMaterial)[0]

        actualStepLength = np.sum(segmentLengths[singleIndices], axis=1)
        #print('Actual step lengths for single material particles:', actualStepLength)
        newPositions[singleIndices] += (actualStepLength[:, None] * newDirections[singleIndices])
        newPositions[singleIndices] = np.round(newPositions[singleIndices], decimals=5)
        #print('New positions:', newPositions[singleIndices])

        energyLossStep = initialEnergies[singleIndices] - sampledEnergies[singleIndices]
        #print('Energy loss step:', energyLossStep)

        newEnergies[singleIndices] = sampledEnergies[singleIndices]
        
        depositEnergy3DStepTraversal(
            voxelIndices=voxelIndices[singleIndices],
            segmentLengths=segmentLengths[singleIndices],
            segmentCounts=segmentCounts[singleIndices],
            energyLossPerStep=energyLossStep,
            energyDepositedVector=energyGrid,
            initialEnergies=initialEnergies[singleIndices],
            fluenceVector=fluenceGrid,
            energyFluenceVector=energyFluenceGrid
        )

        # print('New energies:', newEnergies[singleIndices])
        
        # print('------------------------------------')

    if np.any(crossedMultipleMaterials):
        multipleIndices = np.where(crossedMultipleMaterials)[0]
        #print('Crossed multiple material indices:', multipleIndices)
        
        # Get material indices and segment lengths for multiple materials
        materialsMulti = materialIndices[multipleIndices]
        segmentLengthsMulti = segmentLengths[multipleIndices]
        voxelIndicesMulti = voxelIndices[multipleIndices]
        firstMaterialMulti = firstMaterial[multipleIndices]
        #print('First material:', firstMaterialMulti)
        
        # Create mask: True where material equals firstMaterial, False otherwise
        sameAsFirst = (materialsMulti == firstMaterialMulti)
        lengthInFirstMaterial = np.sum(segmentLengthsMulti * sameAsFirst, axis=1)
        #print('Length in first material:', lengthInFirstMaterial)
        
        energyLossStep = initialEnergies[multipleIndices] - sampledEnergies[multipleIndices]
        fractionStep = lengthInFirstMaterial / stepLength
        realEnergyLossStep = energyLossStep * fractionStep
        finalEnergy = initialEnergies[multipleIndices] - realEnergyLossStep
        #print('Final energies:', finalEnergy)
        
        # Update the postions of the particles
        newPositions[multipleIndices] += lengthInFirstMaterial[:, None] * newDirections[multipleIndices]
        newPositions[multipleIndices] = np.round(newPositions[multipleIndices], decimals=5)
        #print('New positions:', newPositions[multipleIndices])
        
        # Truncate to first-material-only segments
        truncatedSegmentLengths = np.zeros_like(segmentLengthsMulti)
        truncatedVoxelIndices = np.full_like(voxelIndicesMulti, -1)
        truncatedSegmentCounts = np.count_nonzero(sameAsFirst, axis=1)

        truncatedSegmentLengths[sameAsFirst] = segmentLengthsMulti[sameAsFirst]
        truncatedVoxelIndices[sameAsFirst] = voxelIndicesMulti[sameAsFirst]

        depositEnergy3DStepTraversal(
            voxelIndices=truncatedVoxelIndices,
            segmentLengths=truncatedSegmentLengths,
            segmentCounts=truncatedSegmentCounts,
            energyLossPerStep=realEnergyLossStep,
            energyDepositedVector=energyGrid,
            initialEnergies=initialEnergies[multipleIndices],
            fluenceVector=fluenceGrid,
            energyFluenceVector=energyFluenceGrid
        )

        newEnergies[multipleIndices] = finalEnergy
        #print('New energies:', newEnergies[multipleIndices])
        
        #print('------------------------------------')
    
    return newPositions, newDirections, newEnergies

# # --------------- COMMON FUNCTIONS ---------------
@numba.jit(nopython=True, fastmath=True)
def depositEnergy3DStepTraversal(
    voxelIndices, segmentLengths, segmentCounts, energyLossPerStep, energyDepositedVector,
    initialEnergies, fluenceVector, energyFluenceVector, stepLength=1.0
):
    nParticles = segmentCounts.shape[0]
    inverseStepLength = 1.0 / stepLength

    for i in range(nParticles): 
        count = segmentCounts[i]
        if count == 0:
            continue

        # eInstant = initialEnergies[i]
        # eMid = initialEnergies[i] - 0.5 * energyLossPerStep[i]
        eMid = initialEnergies[i] - 0.25 * energyLossPerStep[i]
        for j in range(count):
            frac = segmentLengths[i, j] * inverseStepLength
            deposit = frac * energyLossPerStep[i]

            x, y, z = voxelIndices[i, j]
            energyDepositedVector[x, y, z] += deposit
            fluenceVector[x, y, z] += segmentLengths[i, j]
            # energyFluenceVector[x, y, z] += segmentLengths[i, j] * eInstant
            energyFluenceVector[x, y, z] += segmentLengths[i, j] * eMid

            print(f'Depositing energy: {deposit} at voxel ({x}, {y}, {z})')
            #print(f'Fluence: segment length: {segmentLengths[i,j]} at voxel ({x}, {y}, {z})')
            #print(f'Energy fluence: {segmentLengths[i,j] * eInstant} at voxel ({x}, {y}, {z})')
            

# --------------- COMMON FUNCTIONS ---------------
def simulateBatchParticlesVectorized(
    batchSize, data, gridMap, initialEnergy,
    bigVoxelSize, energyDepositedVector, fluenceVector, 
    energyFluenceVector, cdfs, binEdges, 
    method='transformation',
    angleStep=None, energyStep=None
):
    # Convert to numpy arrays
    bigVoxelSize = np.array(bigVoxelSize)
    gridShape = np.array(gridMap.shape)
    
    # Random -10 mm to +10 mm in X and Y axis
    initialZ = -1.5  # Initial position in mm
    xyRange = 0.  # Half-width in mm (so total 50 mm x 50 mm field)
    
    # Generate uniform random positions in X and Y 
    x = np.random.uniform(-xyRange, xyRange, batchSize)
    y = np.random.uniform(-xyRange, xyRange, batchSize)
    z = np.full(batchSize, initialZ)
    position = np.stack([x, y, z], axis=1)
    
    energy = np.full(batchSize, initialEnergy)
    velocity = np.tile([0.0, 0.0, 1.0], (batchSize, 1))
    active = np.ones(batchSize, dtype=bool)
    # print(f'Initial Energy: {initialEnergy} MeV')
    # print(f'Initial Position: {position}')
    
    # List to store (exit angle, energy) for particles exiting the grid
    exitAngles = np.full(batchSize, np.nan) 
    exitEnergies = np.full(batchSize, np.nan)

    zStep = 1.0  # Fixed step of 1 mm
    while np.any(active):
        energyActive = energy[active]
        
        # Update directions and positions every time a boundary has been crossed or the step length is exceeded
        position[active], velocity[active], energy[active] = scatteringStepMaterial(
            initialPositions=position[active],
            directions=velocity[active],
            initialEnergies=energyActive,
            energyGrid=energyDepositedVector,
            fluenceGrid=fluenceVector,
            energyFluenceGrid=energyFluenceVector,
            grid=gridMap,
            data=data,
            cdfs=cdfs,
            binEdges=binEdges,
            gridShape=gridShape,
            physicalSize=bigVoxelSize,
            method=method,
            angleStep=angleStep,
            energyStep=energyStep,
            stepLength=zStep   
        )
        # print(f'Updated positions:\n{position[active]}')
        # print(f'Updated energies:\n{energy[active]}')
        
        # Check if particles are still active
        withinBounds = np.all(
            (position[active] >= -bigVoxelSize) & (position[active] < bigVoxelSize),
            axis=1
        )
        realEnergiesValid = energy[active] > 0.0
        
        # Determine global indices of active particles
        activeIndices = np.where(active)[0]
        
        # Find those that exited (either out of bounds or energy ≤ 0)
        exitedMask = ~withinBounds & realEnergiesValid
        exitedIndices = activeIndices[exitedMask]

        # Store exit angle and energy
        if exitedIndices.size > 0:
            vz = velocity[active][exitedMask][:, 2]
            vz = np.clip(vz, -1.0, 1.0)  # Safety for arccos
            exitAngles[exitedIndices] = np.degrees(np.arccos(vz))
            exitEnergies[exitedIndices] = energy[active][exitedMask]

        # Update active status
        active[active] = realEnergiesValid & withinBounds

    return exitAngles, exitEnergies
        

       
# --------------- COMMON FUNCTIONS ---------------
def updateDirection(velocity, realAngles):
    numParticles = velocity.shape[0]

    # Sample random azimuthal angles phi ∈ [0, 2π)
    phiAngles = np.random.uniform(0, 2 * np.pi, numParticles)
    thetaAngles = np.radians(realAngles)  # convert degrees to radians

    v = velocity  # shape (numParticles, 3)
    zAxis = np.array([0.0, 0.0, 1.0])

    # Compute perpendicular vector to v to construct rotation plane
    perpVectors = np.cross(v, zAxis)
    perpNorms = np.linalg.norm(perpVectors, axis=1, keepdims=True)

    # Handle near-parallel cases with a fallback axis
    smallMask = (perpNorms[:, 0] < 1e-8)
    perpVectors[smallMask] = np.cross(v[smallMask], np.array([1.0, 0.0, 0.0]))
    perpNorms[smallMask] = np.linalg.norm(perpVectors[smallMask], axis=1, keepdims=True)
    perpVectors /= perpNorms  # normalize perpendicular vector

    # Construct random rotation axis in scattering cone
    cosPhi = np.cos(phiAngles)[:, None]
    sinPhi = np.sin(phiAngles)[:, None]
    rotationAxes = perpVectors * cosPhi + np.cross(v, perpVectors) * sinPhi
    rotationAxes /= np.linalg.norm(rotationAxes, axis=1, keepdims=True)

    # Rodrigues' rotation formula (vectorized)
    cosTheta = np.cos(thetaAngles)[:, None]
    sinTheta = np.sin(thetaAngles)[:, None]

    dotProduct = np.sum(rotationAxes * v, axis=1, keepdims=True)
    rotatedV = (
        v * cosTheta +
        np.cross(rotationAxes, v) * sinTheta +
        rotationAxes * dotProduct * (1 - cosTheta)
    )
    rotatedV /= np.linalg.norm(rotatedV, axis=1, keepdims=True)

    newVelocity = rotatedV

    return newVelocity

# --------------- COMMON FUNCTIONS ---------------
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

# --------------- COMMON FUNCTIONS ---------------
@numba.jit(nopython=True, inline = 'always') 
def getCatmullRomPoints(array, i):
    # array: your pTable or energies
    n = len(array)
    f_1 = array[max(i - 1, 0)]
    f0  = array[i]
    f1  = array[min(i + 1, n - 1)]
    f2  = array[min(i + 2, n - 1)]
    return f_1, f0, f1, f2

# --------------- COMMON FUNCTIONS ---------------
def quadraticInterpolation(x_1, y_1, x0, y0, x1, y1, x):
    L_1 = ((x - x0)*(x - x1)) / ((x_1 - x0)*(x_1 - x1))
    L0 = ((x - x_1)*(x - x1)) / ((x0 - x_1)*(x0 - x1))
    L1 = ((x - x_1)*(x - x0)) / ((x1 - x_1)*(x1 - x0))
    return y_1*L_1 + y0*L0 + y1*L1

# --------------- COMMON FUNCTIONS ---------------
def linearInterpolation(x0, y0, x1, y1, x):
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))

        
# --------------- COMMON FUNCTIONS ---------------
def simulateBatchParticlesWorker(args):
    method = args[29] 
    batchID = args[30]
    
    # Common extraction for all methods
    (
        shm_name, shape, dtype_str,
        shm_cdfs_name, cdfs_shape, cdfs_dtype_str,
        shm_energy_deposited_name, energy_deposited_shape, energy_deposited_dtype_str,
        shm_fluence_name, fluence_shape, fluence_dtype_str,
        shm_energy_fluence_name, energy_fluence_shape, energy_fluence_dtype_str,
        shm_exit_angles_name, exit_angles_shape, exit_angles_dtype_str,
        shm_exit_energies_name, exit_energies_shape, exit_energies_dtype_str,
        batchSize, materials, energies, escapeProbs,
        gridMaterial, initialEnergy,
        bigVoxelSize, seed, _, _  # method and batchID are already extracted
    ) = args[:31]

    # Default to None (for both methods)
    binEdges = angleStep = energyStep = None
    shm_bin_edges_name = bin_edges_shape = bin_edges_dtype_str = None

    # Handle method-specific arguments
    if method == 'normalization':
        (
            shm_bin_edges_name, bin_edges_shape, bin_edges_dtype_str
        ) = args[31:]

    elif method == 'transformation':
        (
            binEdges, angleStep, energyStep
        ) = args[31:]

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
    shared_energyDeposited = np.ndarray(energy_deposited_shape, dtype=np.dtype(energy_deposited_dtype_str), buffer=existing_shm_energy_deposited.buf)

    # Attach to shared memory for fluence and energy fluence
    existing_shm_fluence = shared_memory.SharedMemory(name=shm_fluence_name)
    shared_fluence = np.ndarray(fluence_shape, dtype=np.dtype(fluence_dtype_str), buffer=existing_shm_fluence.buf)
    
    existing_shm_energy_fluence = shared_memory.SharedMemory(name=shm_energy_fluence_name)
    shared_energyFluence = np.ndarray(energy_fluence_shape, dtype=np.dtype(energy_fluence_dtype_str), buffer=existing_shm_energy_fluence.buf)
    
    # Attach to shared memory for exit angles and energies
    existing_shm_exit_angles = shared_memory.SharedMemory(name=shm_exit_angles_name)
    shared_exit_angles = np.ndarray(exit_angles_shape, dtype=np.dtype(exit_angles_dtype_str), buffer=existing_shm_exit_angles.buf)

    existing_shm_exit_energies = shared_memory.SharedMemory(name=shm_exit_energies_name)    
    shared_exit_energies = np.ndarray(exit_energies_shape, dtype=np.dtype(exit_energies_dtype_str), buffer=existing_shm_exit_energies.buf)

    if method == 'normalization':
        existing_shm_bin_edges = shared_memory.SharedMemory(name=shm_bin_edges_name)
        binEdges = np.ndarray(bin_edges_shape, dtype=np.dtype(bin_edges_dtype_str), buffer=existing_shm_bin_edges.buf)

    # Initialize local vectors
    localEnergyDeposited = np.zeros_like(shared_energyDeposited)
    localFluence = np.zeros_like(shared_fluence)
    localEnergyFluence = np.zeros_like(shared_energyFluence)
    
    # Prepare data for simulation
    data = {
        'probTable': probTable,
        'materials': materials,
        'energies': energies,
        'escapeProbs': escapeProbs
    }

    # Run the simulation
    result = simulateBatchParticlesVectorized(
        batchSize=batchSize,
        data=data,
        gridMap=gridMaterial,
        initialEnergy=initialEnergy,
        bigVoxelSize=bigVoxelSize,
        energyDepositedVector=localEnergyDeposited,
        fluenceVector=localFluence,
        energyFluenceVector=localEnergyFluence,
        cdfs=cdfs,
        binEdges=binEdges,
        method=method,
        angleStep=angleStep,
        energyStep=energyStep
    )

    # Safely merge local vectors into shared memory (atomic if required)
    np.add(shared_energyDeposited, localEnergyDeposited, out=shared_energyDeposited)
    np.add(shared_fluence, localFluence, out=shared_fluence)
    np.add(shared_energyFluence, localEnergyFluence, out=shared_energyFluence)
    
    # Calculate global start/end indices for this batch's particles
    startIdx = batchID * batchSize
    endIdx = startIdx + batchSize
    
    # Save exit angles and energies in shared arrays at the right positions
    exitAngles, exitEnergies = result  # unpack tuple
    shared_exit_angles[startIdx:endIdx] = exitAngles
    shared_exit_energies[startIdx:endIdx] = exitEnergies
    
    # Optional: logging
    totalEnergy = np.sum(localEnergyDeposited)
    with open(f"logs{method}.txt", "a") as log_file:
        log_file.write(f"Batch {batchID:04d} | Deposited Energy: {totalEnergy:.2f} MeV\n")

    return result

# --------------- COMMON FUNCTIONS ---------------
def runMultiprocessedBatchedSim(
    totalSamples, batchSize, numWorkers,
    shm_prob_table, prob_table_shape, prob_table_dtype,
    shm_cdfs, cdfs_shape, cdfs_dtype,
    shm_energy_deposited, energy_deposited_shape, energy_deposited_dtype,
    shm_fluence, fluence_shape, fluence_dtype,
    shm_energy_fluence, energy_fluence_shape, energy_fluence_dtype,
    shm_exit_angles, exit_angles_shape, exit_angles_dtype,
    shm_exit_energies, exit_energies_shape, exit_energies_dtype,
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
    
    baseSeed = 456789
    seedSequence = np.random.SeedSequence(baseSeed)
    childSeeds = seedSequence.spawn(numBatches)
    
    # Create a list of arguments for each worker with chunked tasks
    for i in range(numBatches):
        currentSeed = childSeeds[i].generate_state(1)[0]
        
        args = [
            shm_prob_table.name, prob_table_shape, prob_table_dtype.name,
            shm_cdfs.name, cdfs_shape, cdfs_dtype.name,
            shm_energy_deposited.name, energy_deposited_shape, energy_deposited_dtype.name,
            shm_fluence.name, fluence_shape, fluence_dtype.name,
            shm_energy_fluence.name, energy_fluence_shape, energy_fluence_dtype.name,
            shm_exit_angles.name, exit_angles_shape, exit_angles_dtype.name,
            shm_exit_energies.name, exit_energies_shape, exit_energies_dtype.name,
            min(batchSize, totalSamples - i * batchSize),
            data['materials'], data['energies'], data['escapeProbs'],
            gridMaterial, initialEnergy,
            bigVoxelSize, currentSeed, # seed for each batch, reproducible
            method,
            i # batch index
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

def sampleAndPlotSingleEnergy(
    data,
    cdfs,
    binEdges,
    method,
    materialIndex,
    energy,
    saveFig = False,
    numSamples=100_000,
    angleStep=None,
    energyStep=None,
    ):
    # Create input arrays
    materials = np.full(numSamples, materialIndex, dtype=int)
    energies = np.full(numSamples, energy, dtype=float)

    # Sample angles and energy losses
    sampledAngles, sampledEnergies = sampleFromCDF(
        data=data,
        materialsIndex=materials,
        energies=energies,
        cdfs=cdfs,
        binEdges=binEdges,
        method=method,
        angleStep=angleStep,
        energyStep=energyStep
    )   

    # --- ANGLE HISTOGRAM ---
    angleHist, angleBins = np.histogram(sampledAngles, bins=100)
    angleCenters = 0.5 * (angleBins[:-1] + angleBins[1:])
    angleHistNorm = angleHist / np.max(angleHist)

    # Threshold: 1/100 of the max
    threshold = 1 / 100
    aboveThreshold = np.where(angleHistNorm <= threshold)[0]
    if len(aboveThreshold) > 0:
        thresholdAngle = angleCenters[aboveThreshold[0]]
    else:
        thresholdAngle = None

    plt.figure(figsize=(7.25, 6))
    plt.plot(angleCenters, angleHistNorm, color='darkred')
    plt.xlabel("Angle (º)")
    plt.ylabel("Normalized Frequency (max=1)")
    plt.title(f"Angle Sampling - Material {materialIndex}, E = {energy} MeV, Method: {method}")
    if thresholdAngle is not None:
        plt.axvline(thresholdAngle, color='blue', linestyle='--', label=f"1% max at {thresholdAngle:.2f}°", linewidth=0.5)
        plt.legend()
    plt.tight_layout()
    if saveFig:
        plt.savefig(f"SamplingAnglesNorm{method}_G{materialIndex}_ENERGY{energy}.pdf")
    plt.close()

    # --- ENERGY HISTOGRAM ---
    energyHist, energyBins = np.histogram(sampledEnergies, bins=100)
    energyCenters = 0.5 * (energyBins[:-1] + energyBins[1:])
    energyHistNorm = energyHist / np.max(energyHist)

    aboveThresholdE = np.where(energyHistNorm <= threshold)[0]
    if len(aboveThresholdE) > 0:
        thresholdEnergy = energyCenters[aboveThresholdE[0]]
    else:
        thresholdEnergy = None

    plt.figure(figsize=(7.25, 6))
    plt.plot(energyCenters, energyHistNorm, color='darkred')
    plt.xlabel("Final energy (MeV)")
    plt.ylabel("Normalized Frequency (max=1)")
    plt.title(f"Energy Sampling - Material {materialIndex}, E = {energy} MeV, Method: {method}")
    if thresholdEnergy is not None:
        plt.axvline(thresholdEnergy, color='blue', linestyle='--', label=f"1% max at {thresholdEnergy:.2f} MeV", linewidth=0.5)
        plt.legend()
    plt.tight_layout()
    if saveFig:
        plt.savefig(f"SamplingEnergyNorm{method}_G{materialIndex}_ENERGY{energy}.pdf")
    plt.close()

    # --- STATISTICS ---
    print(f'Mean energy: {np.mean(sampledEnergies):.4f}')
    print(f'Std energy:  {np.std(sampledEnergies):.4f}')
    print(f'Mean angle:  {np.mean(sampledAngles):.4f}')
    print(f'Std angle:   {np.std(sampledAngles):.4f}')
    if thresholdAngle is not None:
        print(f"Angle at which freq drops to 1%: {thresholdAngle:.2f}°")
    if thresholdEnergy is not None:
        print(f"Energy at which freq drops to 1%: {thresholdEnergy:.2f} MeV")

def plotExitAngleEnergyHistograms(
    sampledAngles,
    sampledEnergies,
    energy,
    method,
    numberOfBins,
    threshold=0.01,
    saveFig=True
):
    # --- ANGLE HISTOGRAM ---
    angleHist, angleBins = np.histogram(sampledAngles, bins=numberOfBins)
    angleCenters = 0.5 * (angleBins[:-1] + angleBins[1:])
    angleHistNorm = angleHist / np.max(angleHist)

    aboveThresholdAngle = np.where(angleHistNorm <= threshold)[0]
    thresholdAngle = angleCenters[aboveThresholdAngle[0]] if len(aboveThresholdAngle) > 0 else None

    plt.figure(figsize=(7.25, 6))
    plt.plot(angleCenters, angleHistNorm, color='darkred')
    plt.xlabel("Angle (º)")
    plt.ylabel("Normalized Frequency (max=1)")
    plt.title(f"Angle Sampling - E = {energy} MeV, Method: {method}")
    if thresholdAngle is not None:
        plt.axvline(thresholdAngle, color='blue', linestyle='--', label=f"1% max at {thresholdAngle:.2f}°", linewidth=0.5)
        plt.legend()
    plt.tight_layout()
    if saveFig:
        plt.savefig(f"SamplingAnglesNorm{method}_ENERGY{energy}.pdf")
    plt.close()

    # --- ENERGY HISTOGRAM ---
    energyHist, energyBins = np.histogram(sampledEnergies, bins=numberOfBins)
    energyCenters = 0.5 * (energyBins[:-1] + energyBins[1:])
    energyHistNorm = energyHist / np.max(energyHist)

    aboveThresholdEnergy = np.where(energyHistNorm <= threshold)[0]
    thresholdEnergy = energyCenters[aboveThresholdEnergy[0]] if len(aboveThresholdEnergy) > 0 else None

    plt.figure(figsize=(7.25, 6))
    plt.plot(energyCenters, energyHistNorm, color='darkred')
    plt.xlabel("Final energy (MeV)")
    plt.ylabel("Normalized Frequency (max=1)")
    plt.title(f"Energy Sampling - E = {energy} MeV, Method: {method}")
    if thresholdEnergy is not None:
        plt.axvline(thresholdEnergy, color='blue', linestyle='--', label=f"1% max at {thresholdEnergy:.2f} MeV", linewidth=0.5)
        plt.legend()
    plt.tight_layout()
    if saveFig:
        plt.savefig(f"SamplingEnergyNorm{method}_ENERGY{energy}.pdf")
    plt.close()

    # --- STATISTICS ---
    print(f'Mean energy: {np.mean(sampledEnergies):.4f}')
    print(f'Std energy:  {np.std(sampledEnergies):.4f}')
    print(f'Mean angle:  {np.mean(sampledAngles):.4f}')
    print(f'Std angle:   {np.std(sampledAngles):.4f}')
    if thresholdAngle is not None:
        print(f"Angle at which freq drops to 1%: {thresholdAngle:.2f}°")
    if thresholdEnergy is not None:
        print(f"Energy at which freq drops to 1%: {thresholdEnergy:.2f} MeV")

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
    samplingN = 1
    initialEnergy = 50.  # MeV
    bigVoxelSize = np.array((0.5, 0.5, 1.5), dtype=np.float32) # in mm
    voxelShapeBins = np.array((1, 1, 3), dtype=np.int32)
    voxelSize = 2 * bigVoxelSize / voxelShapeBins # in mm 
    
    angleRange = (0, 200)
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
            'npzPath': './Table/4DTableTransSphere.npz',
            'npyPath': './Numpy/',
            'csvPath': './CSV/'
        },
        'normalization': {
            'npzPath': './TableNormalized/4DTableNormSphere.npz',
            'npyPath': './NumpyNormalized/',
            'csvPath': './CSVNormalized/'
        }
    }

    npzPath = baseFolder[method]['npzPath']
    npyPath = baseFolder[method]['npyPath']
    csvPath = baseFolder[method]['csvPath']

    # Make sure the timing directory exists
    Path(npyPath).mkdir(parents=True, exist_ok=True)
    Path(csvPath).mkdir(parents=True, exist_ok=True)

    # --- Load table ---
    rawData = np.load(npzPath, mmap_mode='r', allow_pickle=True)
    probTable = rawData['probTable']
    materials = rawData['materials'].tolist()
    energies = rawData['energies']
    
    # --- Load braggTable ---
    braggPath = './BraggPeak/escapeProbs.npz'
    braggData = np.load(braggPath)
    escapeProbsData = {int(k): braggData[k] for k in braggData.files}

    # Share data fields
    data = {
        'probTable': probTable,
        'materials': materials,
        'energies': energies,
        'escapeProbs': escapeProbsData
    }

    if method == 'normalization':
        # Load extra arrays only used in normalization
        data['thetaMax'] = rawData['thetaMax']
        data['thetaMin'] = rawData['thetaMin']
        data['energyMin'] = rawData['energyMin']
        data['energyMax'] = rawData['energyMax']
        
    rawData.close()
    braggData.close()
    
    binEdges = None
    
    # Build CDFs 
    if method == 'normalization':
        cdfs, binEdges = buildCdfsAndCompactBins(data=data)      
        # Remove unnecessary keys from data to free memory
        for key in ['thetaMax', 'thetaMin', 'energyMin', 'energyMax']:
            del data[key]
    else:
        cdfs = buildCdfsFromProbTable(probTable)

    # Shared memory for probability table and CDFs
    energyDeposited = np.zeros(voxelShapeBins, dtype=np.float32)
    shm_prob_table, prob_table_shape, prob_table_dtype = createSharedMemory(probTable)
    shm_cdfs, cdfs_shape, cdfs_dtype = createSharedMemory(cdfs)
    shm_energy_deposited, energyDeposited_shape, energyDeposited_dtype = createSharedMemory(energyDeposited)
    
    # Shared memory for fluence and energyFluence
    fluence = np.zeros(voxelShapeBins, dtype=np.float32)
    energyFluence = np.zeros(voxelShapeBins, dtype=np.float32)
    shm_fluence, fluence_shape, fluence_dtype = createSharedMemory(fluence)
    shm_energy_fluence, energyFluence_shape, energyFluence_dtype = createSharedMemory(energyFluence)
    
    # Shared memory for exit angles and energies
    exitAngles = np.full((samplingN,), np.nan, dtype=np.float32)
    exitEnergies = np.full((samplingN,), np.nan, dtype=np.float32)
    shm_exit_angles, exit_angles_shape, exit_angles_dtype = createSharedMemory(exitAngles)
    shm_exit_energies, exit_energies_shape, exit_energies_dtype = createSharedMemory(exitEnergies)
    
    batchSize = 10_000
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
        
    # sampleAndPlotSingleEnergy(
    #     data=data,
    #     cdfs=cdfs,
    #     binEdges=binEdges,
    #     method=method,
    #     materialIndex=0,  # Example material index
    #     energy=initialEnergy,  # Example energy
    #     saveFig=True,  # Save the plots
    #     numSamples=10_000_000  # Number of samples for plotting
    # )
        
    # Run simulation
    print(f"Running simulation using '{method}' method.")
    print(f"Characteristics of the simulation:")
    print(f"Voxel size: {voxelSize[0]} x {voxelSize[1]} x {voxelSize[2]} mm")
    print(f"Voxel shape bins: {voxelShapeBins[0]} x {voxelShapeBins[1]} x {voxelShapeBins[2]}")
    print(f"Number of workers: {numWorkers}")
    print(f"Batch size: {batchSize}")
    print(f"Number of sampling points: {samplingN}")

    runMultiprocessedBatchedSim(
        samplingN, batchSize, numWorkers,
        shm_prob_table, prob_table_shape, prob_table_dtype,
        shm_cdfs, cdfs_shape, cdfs_dtype,
        shm_energy_deposited, energyDeposited_shape, energyDeposited_dtype,
        shm_fluence, fluence_shape, fluence_dtype,
        shm_energy_fluence, energyFluence_shape, energyFluence_dtype,
        shm_exit_angles, exit_angles_shape, exit_angles_dtype,
        shm_exit_energies, exit_energies_shape, exit_energies_dtype,
        data, gridMaterial, initialEnergy,
        bigVoxelSize, method, 
        **kwargs
    )
                    
    energyVector3D = np.ndarray(energyDeposited.shape, dtype=energyDeposited.dtype, buffer=shm_energy_deposited.buf).copy()
    np.save(Path(npyPath) / f'energyDeposited{method}.npy', energyVector3D)
    
    # Make a copy of exit angles and energies
    exitAnglesVector = np.ndarray(exitAngles.shape, dtype=exitAngles.dtype, buffer=shm_exit_angles.buf).copy()
    exitEnergiesVector = np.ndarray(exitEnergies.shape, dtype=exitEnergies.dtype, buffer=shm_exit_energies.buf).copy()
    plotExitAngleEnergyHistograms(
        sampledAngles=exitAnglesVector,
        sampledEnergies=exitEnergiesVector,
        energy=initialEnergy,
        method=method,
        numberOfBins=angleBins
    )

    # First same units than topas
    # Fluence 1 / mm^2, energyFluence MeV / mm^2
    volumeVoxel = voxelSize[0] * voxelSize[1] * voxelSize[2]
    fluenceVector3D = np.ndarray(fluence.shape, dtype=fluence.dtype, buffer=shm_fluence.buf).copy()
    fluenceVector3D = fluenceVector3D / volumeVoxel
    
    energyFluenceVector3D = np.ndarray(energyFluence.shape, dtype=energyFluence.dtype, buffer=shm_energy_fluence.buf).copy()
    energyFluenceVector3D = energyFluenceVector3D / volumeVoxel
    
    fluenceThreshold = 1e-6
    mask = fluenceVector3D > fluenceThreshold
    meanEnergyGrid = np.zeros_like(fluenceVector3D)
    meanEnergyGrid[mask] = energyFluenceVector3D[mask] / fluenceVector3D[mask]
    # profileZMean = meanEnergyGrid[:, 5, 0]
    # print(profileZMean)
    
    np.save(Path(npyPath) / f'meanEnergyGrid{method}.npy', meanEnergyGrid)
         
    totalEnergy = energyVector3D.sum()
    print(f"Total energy: {totalEnergy:.6f} MeV")
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
    
    shm_fluence.close()
    shm_fluence.unlink()
    shm_energy_fluence.close()
    shm_energy_fluence.unlink()
    
    shm_exit_angles.close()
    shm_exit_angles.unlink()
    shm_exit_energies.close()
    shm_exit_energies.unlink()
    
    if method == 'normalization':
        shm_bin_edges.unlink()
        shm_bin_edges.close()
        
    endTime = time.time()
    print(f"Simulation time: {endTime - startTime:.12f} seconds")
    print()