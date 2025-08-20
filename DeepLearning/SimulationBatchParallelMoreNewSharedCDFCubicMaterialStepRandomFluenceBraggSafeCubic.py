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
import pydicom
import psutil

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
@numba.jit(parallel=True)
def buildCdfsFromProbTable(probTable, availableEnergies):
    numMaterials, numEnergies, angleBins, energyBins = probTable.shape
    totalBins = angleBins * energyBins

    cdfs = np.empty((numMaterials, numEnergies, totalBins), dtype=np.float32)
    minEnergyByMaterial = np.full(numMaterials, -1.0, dtype=np.float32)

    for m in prange(numMaterials): 
        
        # We need to find the index of the lowest energy with a non-zero CDF.
        lastNonZeroCdfEnergyIndex = -1
        
        for e in range(numEnergies):
            flatProbTable = probTable[m, e].ravel()  # flatten 2D bin table
            cdf = np.empty_like(flatProbTable)
            total = 0.0
            for i in range(totalBins):  # compute cumulative sum manually
                total += flatProbTable[i]
                cdf[i] = total
            norm = cdf[-1]
            
            # Find the last energy with a non-zero CDF
            if norm > 0.0:
                lastNonZeroCdfEnergyIndex = e
            
            if norm == 0.0:  # handle zero norm case
                # Set CDF to zeros if total probability is zero (expected for some low energies)
                cdfs[m, e, :] = 0
            else:
                for i in range(totalBins):  # normalize manually
                    cdf[i] /= norm
                    cdfs[m, e, i] = cdf[i]
    
        # Find the lowest energy with a non-zero CDF, the else statement should never trigger
        if lastNonZeroCdfEnergyIndex != -1:
            minEnergyByMaterial[m] = availableEnergies[lastNonZeroCdfEnergyIndex]
        else:
            print(f"[ERROR] Warning: No non-zero CDF values found for material {m} for transformation variable. Setting minEnergyByMaterial to 5.0 MeV.")
            minEnergyByMaterial[m] = 5.0 # MeV

    return cdfs, minEnergyByMaterial
                
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
    
    # Material-specific minimum valid energy (indexed by material index)
    minEnergyByMaterial = data['minEnergyByMaterial']
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
    #print('Closest energy indices:', closestEnergyIndices)
    
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
        #print('materialIdx:', materialIdx, 'energyIdx:', energyIdx)
        sampleIndices = np.where((materials == materialIdx) & (closestEnergyIndices == energyIdx))[0]
        #print('Sample indices:', sampleIndices)
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
    
    availableEnergies = data['energies']
    
    numMaterials, numEnergies, angleBins, energyBins = probTable.shape
    totalBins = angleBins * energyBins
    maxBinCount = max(angleBins + 1, energyBins + 1)

    cdfs = np.zeros((numMaterials, numEnergies, totalBins), dtype=np.float32)
    # Unified bin edges: [m, e, 0] = angleEdges, [m, e, 1] = energyEdges
    binEdges = np.zeros((numMaterials, numEnergies, 2, maxBinCount), dtype=np.float32)
    minEnergyByMaterial = np.full(numMaterials, -1.0, dtype=np.float32)
    
    # Precompute normalized linspaces for re-use
    normAngle = np.linspace(0, 1, angleBins + 1, dtype=np.float32)
    normEnergy = np.linspace(0, 1, energyBins + 1, dtype=np.float32)

    for m in range(numMaterials):
        lastNonZeroCdfEnergyIndex = -1
        
        for e in range(numEnergies):
            hist = probTable[m, e].reshape(-1)
            total = hist.sum(dtype=np.float32)

            if total > 0.0:
                lastNonZeroCdfEnergyIndex = e
                cdfs[m, e] = np.cumsum(hist, dtype=np.float32) / total

            thetaRange = thetaMax[m, e] - thetaMin[m, e]
            energyRange = energyMax[m, e] - energyMin[m, e]
            
            # Slicing, although the bins are the same size for all materials and energies
            binEdges[m, e, 0, :angleBins + 1] = thetaMin[m, e] + thetaRange * normAngle
            binEdges[m, e, 1, :energyBins + 1] = energyMin[m, e] + energyRange * normEnergy
            
        # Find the lowest energy with a non-zero CDF, the else statement should never trigger
        if lastNonZeroCdfEnergyIndex != -1:
            minEnergyByMaterial[m] = availableEnergies[lastNonZeroCdfEnergyIndex]
        else:
            print(f"[ERROR] Warning: No non-zero CDF values found for material {m} for transformation variable. Setting minEnergyByMaterial to 5.0 MeV.")
            minEnergyByMaterial[m] = 5.0 # MeV
            
    return cdfs, binEdges, minEnergyByMaterial

# def sampleFromCDFVectorizedNormalizationNearest(
#     data: dict, 
#     materials: np.ndarray, 
#     energies: np.ndarray, 
#     cdfs: np.ndarray, 
#     binEdges: np.ndarray
# ) -> Tuple[np.ndarray, np.ndarray]:
    
#     availableEnergies = data['energies']
#     angleBins, energyBins = data['probTable'].shape[2:]
    
#     # Material-specific minimum valid energy (indexed by material index)
#     minEnergyByMaterial = data['minEnergyByMaterial']
#     materialMinEnergies = minEnergyByMaterial[materials]

#     minEnergy = np.min(availableEnergies)
#     maxEnergy = np.max(availableEnergies)
#     clippedEnergies = np.clip(energies, minEnergy, maxEnergy)

#     # --- Use the more efficient searchsorted method to find the nearest energy index ---
#     reversedEnergies = availableEnergies[::-1]
#     insertPos = np.searchsorted(reversedEnergies, clippedEnergies, side='left')
#     insertPos = np.clip(insertPos, 1, len(reversedEnergies) - 1)

#     left = reversedEnergies[insertPos - 1]
#     right = reversedEnergies[insertPos]
#     chooseLeft = np.abs(clippedEnergies - left) < np.abs(clippedEnergies - right)
    
#     closestEnergyIndices = len(availableEnergies) - 1 - np.where(chooseLeft, insertPos - 1, insertPos)
#     # ----------------------------------------------------------------------------------

#     rand = np.random.random(size=energies.shape).astype(np.float32)

#     sampledAngles = np.zeros_like(energies, dtype=np.float32)
#     sampledEnergies = np.zeros_like(energies, dtype=np.float32)

#     uniquePairs = set(zip(materials, closestEnergyIndices))
    
#     for materialIdx, energyIdx in uniquePairs:
#         idxs = np.where((materials == materialIdx) & (closestEnergyIndices == energyIdx))[0].astype(np.int32)
        
#         angleEdges = binEdges[materialIdx, energyIdx, 0, :angleBins + 1]
#         energyEdges = binEdges[materialIdx, energyIdx, 1, :energyBins + 1]

#         if angleEdges[0] == angleEdges[-1] or energyEdges[0] == energyEdges[-1]:
#             sampledAngles[idxs] = 0.0
#             sampledEnergies[idxs] = 0.0
#             continue

#         angleStep = angleEdges[1] - angleEdges[0]
#         energyStep = energyEdges[1] - energyEdges[0]

#         sampleCDFForEnergyGroup(
#             materialIdx, energyIdx, idxs, rand[idxs],
#             cdfs, angleBins, energyBins, angleEdges, energyEdges,
#             angleStep, energyStep, sampledAngles, sampledEnergies
#         )

#     # Apply the mask for energies below the material-specific minimum
#     mask = energies < materialMinEnergies
#     sampledAngles = np.where(mask, 0.0, sampledAngles)
#     sampledEnergies = np.where(mask, 0.0, sampledEnergies)

#     return sampledAngles, sampledEnergies

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
    minEnergyByMaterial = data['minEnergyByMaterial']
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

    interpolateMask = energies > materialMinEnergies
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


# --------------- LINEAR INTERPOLATION ---------------
def sampleFromCDFVectorizedNormalizationLinear(
    data: dict, 
    materials: np.ndarray, 
    energies: np.ndarray, 
    cdfs: np.ndarray, 
    binEdges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    
    availableEnergies = data['energies']
    angleBins, energyBins = data['probTable'].shape[2:]
    
    minEnergyByMaterial = data['minEnergyByMaterial']
    materialMinEnergies = minEnergyByMaterial[materials]

    minEnergy = np.min(availableEnergies)
    maxEnergy = np.max(availableEnergies)
    clippedEnergies = np.clip(energies, minEnergy, maxEnergy)

    # Find the two nearest energy indices
    reversedEnergies = availableEnergies[::-1]
    insertPos = np.searchsorted(reversedEnergies, clippedEnergies, side='left')
    insertPos = np.clip(insertPos, 1, len(reversedEnergies) - 1)

    base = len(availableEnergies) - 1
    i1 = base - insertPos
    i0 = np.clip(i1 - 1, 0, base)
    
    e0 = availableEnergies[i0]
    e1 = availableEnergies[i1]
    
    # Calculate linear interpolation weights
    weights = np.where(
        e1 != e0,
        (clippedEnergies - e0) / (e1 - e0),
        0.0
    ).astype(np.float32)

    rand = np.random.random(size=energies.shape).astype(np.float32)

    out = {}
    for label, indices in zip(['0', '1'], [i0, i1]):
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

    sampledAngles = out['0_angles'] * (1 - weights) + out['1_angles'] * weights
    sampledEnergies = out['0_energies'] * (1 - weights) + out['1_energies'] * weights

    # Apply the mask
    mask = energies < materialMinEnergies
    sampledAngles = np.where(mask, 0.0, sampledAngles)
    sampledEnergies = np.where(mask, 0.0, sampledEnergies)

    return sampledAngles, sampledEnergies

# --------------- IMPROVED NEAREST NEIGHBOR WITH CORRECTION ---------------
def sampleFromCDFVectorizedNormalizationNearest(
    data: dict, 
    materials: np.ndarray, 
    energies: np.ndarray, 
    cdfs: np.ndarray, 
    binEdges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    
    availableEnergies = data['energies']
    angleBins, energyBins = data['probTable'].shape[2:]
    
    # Material-specific minimum valid energy (indexed by material index)
    minEnergyByMaterial = data['minEnergyByMaterial']
    materialMinEnergies = minEnergyByMaterial[materials]

    minEnergy = np.min(availableEnergies)
    maxEnergy = np.max(availableEnergies)
    clippedEnergies = np.clip(energies, minEnergy, maxEnergy)

    # --- Use the more efficient searchsorted method to find the nearest energy index ---
    reversedEnergies = availableEnergies[::-1]
    insertPos = np.searchsorted(reversedEnergies, clippedEnergies, side='left')
    insertPos = np.clip(insertPos, 1, len(reversedEnergies) - 1)

    left_idx = len(availableEnergies) - 1 - (insertPos)
    right_idx = len(availableEnergies) - 1 - (insertPos - 1)
    
    left_energy = availableEnergies[left_idx]
    right_energy = availableEnergies[right_idx]
    
    chooseRight = np.abs(clippedEnergies - right_energy) < np.abs(clippedEnergies - left_energy)
    
    closestEnergyIndices = np.where(chooseRight, right_idx, left_idx)
    closestEnergy = availableEnergies[closestEnergyIndices]
    # ----------------------------------------------------------------------------------

    rand = np.random.random(size=energies.shape).astype(np.float32)

    sampledAngles = np.zeros_like(energies, dtype=np.float32)
    sampledEnergies = np.zeros_like(energies, dtype=np.float32)

    uniquePairs = set(zip(materials, closestEnergyIndices))
    
    for materialIdx, energyIdx in uniquePairs:
        idxs = np.where((materials == materialIdx) & (closestEnergyIndices == energyIdx))[0].astype(np.int32)
        
        angleEdges = binEdges[materialIdx, energyIdx, 0, :angleBins + 1]
        energyEdges = binEdges[materialIdx, energyIdx, 1, :energyBins + 1]

        if angleEdges[0] == angleEdges[-1] or energyEdges[0] == energyEdges[-1]:
            sampledAngles[idxs] = 0.0
            sampledEnergies[idxs] = 0.0
            continue

        angleStep = angleEdges[1] - angleEdges[0]
        energyStep = energyEdges[1] - energyEdges[0]

        sampleCDFForEnergyGroup(
            materialIdx, energyIdx, idxs, rand[idxs],
            cdfs, angleBins, energyBins, angleEdges, energyEdges,
            angleStep, energyStep, sampledAngles, sampledEnergies
        )

    # --- Apply a linear correction to the sampled results ---
    correction_factor = 0.5 
    
    energy_difference = clippedEnergies - closestEnergy
    
    sampledAngles += energy_difference * correction_factor
    sampledEnergies += energy_difference * correction_factor
    # --------------------------------------------------------

    # Apply the mask for energies below the material-specific minimum
    mask = energies < materialMinEnergies
    sampledAngles = np.where(mask, 0.0, sampledAngles)
    sampledEnergies = np.where(mask, 0.0, sampledEnergies)

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


@numba.jit(nopython=True, inline='always')
def sampleCDFForEnergyGroup(materialIdx, energyIdx, sampleIndices, randValues, cdfs,
                            angleBins, energyBins, angleEdges, energyEdges, angleStep, energyStep,
                            sampledAngles, sampledEnergies):
    
    cdf = cdfs[materialIdx, energyIdx]

    for i in range(sampleIndices.size):
        idx = sampleIndices[i]
        r = randValues[i]

        flatIdx = np.searchsorted(cdf, r)
        if flatIdx >= cdf.size:
            flatIdx = cdf.size - 1

        # Handle edge case for the first bin
        if flatIdx == 0:
            cdf_start = 0.0
            cdf_end = cdf[flatIdx]
        else:
            cdf_start = cdf[flatIdx - 1]
            cdf_end = cdf[flatIdx]

        # Get the corresponding bin indices
        angleIdx = flatIdx // energyBins
        energyIdxLocal = flatIdx % angleBins

        # Get the physical bin edges
        angle_start = angleEdges[angleIdx]
        angle_end = angleEdges[angleIdx + 1]
        energy_start = energyEdges[energyIdxLocal]
        energy_end = energyEdges[energyIdxLocal + 1]

        # Calculate the fraction of the way through the bin
        cdf_range = cdf_end - cdf_start
        if cdf_range > 0:
            frac = (r - cdf_start) / cdf_range
        else:
            frac = 0.0

        # Perform linear interpolation to get the final value
        angle = angle_start + frac * (angle_end - angle_start)
        energy = energy_start + frac * (energy_end - energy_start)

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
def computeIndices(
    positions : np.ndarray,
    shape : tuple,
    physSize: tuple,
    grid : np.ndarray = None,
    returnMaterial : bool = False
    ):
    """
    Map world positions to voxel indices (and optionally material IDs).

    Parameters
    ----------
    positions : (N,3) float32 – world coordinates
    shape     : (3,) int32    – (X, Y, Z) voxel counts
    physSize  : (3,) float32  – CT physical size (mm)
    grid      : (X, Y, Z) int32 – material IDs (if requested)
    returnMaterial : bool

    Returns
    -------
    ix, iy, iz : (N,) int32 indices
    matID      : (N,) int32 material IDs (if requested)
    """
    # Precompute scale factors
    scale0 = shape[0] * 0.5 / physSize[0]
    scale1 = shape[1] * 0.5 / physSize[1]
    scale2 = shape[2] * 0.5 / physSize[2]

    # Bulk index computation
    ix = ((positions[:, 0] + physSize[0]) * scale0).astype(np.int32)
    iy = ((positions[:, 1] + physSize[1]) * scale1).astype(np.int32)
    iz = ((positions[:, 2] + physSize[2]) * scale2).astype(np.int32)

    # Clamp indices to the valid [0, shape-1] range
    np.clip(ix, 0, shape[0]-1, out=ix)
    np.clip(iy, 0, shape[1]-1, out=iy)
    np.clip(iz, 0, shape[2]-1, out=iz)

    #  Optional material lookup
    if returnMaterial and grid is not None:
        # Flatten the 3‑D grid
        grid_flat = grid.ravel(order='C')         
        stride_xy = shape[1] * shape[2]
        stride_x  = shape[2]
        flat = ix * stride_xy + iy * stride_x + iz
        matID = grid_flat[flat]       
        return matID

    return ix, iy, iz

# --------------- COMMON FUNCTIONS ---------------
def getMaterialIndex(HU_vals: np.ndarray, CT_data: np.ndarray) -> np.ndarray:
    """
    Vectorised helper – works on *exact* HU values (or the nearest lower one).

    Parameters
    ----------
    HU_vals   : 1‑D, sorted ascending, dtype int32 or float32
    CT_data   : any‑dimensional array that contains the same values that appear
                in HU_vals (or values that fall inside the range).

    Returns
    -------
    mat_index : same shape as CT_data, dtype=np.int32
    """
    flat = CT_data.ravel()
    idx  = np.searchsorted(HU_vals, flat, side='right') - 1
    idx  = np.clip(idx, 0, len(HU_vals)-1)
    return idx.reshape(CT_data.shape).astype(np.int32)

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
        # return sampleFromCDFVectorizedNormalizationCubic(
        #     data=data,
        #     materials=materialsIndex,
        #     energies=energies,
        #     cdfs=cdfs,
        #     binEdges=binEdges
        # )
        # return sampleFromCDFVectorizedNormalizationLinear(
        #     data=data,
        #     materials=materialsIndex,
        #     energies=energies,
        #     cdfs=cdfs,
        #     binEdges=binEdges
        # )
        return sampleFromCDFVectorizedNormalizationNearest(
            data=data,
            materials=materialsIndex,
            energies=energies,
            cdfs=cdfs,
            binEdges=binEdges
        )
    else:
        raise ValueError(f"Unknown sampling method: {method}")

# --------------- COMMON FUNCTIONS ---------------
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
    debug,
    binEdges,
    gridShape,
    physicalSize,
    method,
    angleStep,
    energyStep,
):
    newPositions = initialPositions.copy()
    newDirections = directions.copy()
    newEnergies = initialEnergies.copy()
    
    # Get material index
    rawHUvalues = computeIndices(
        positions=initialPositions,
        shape=gridShape,
        physSize=physicalSize,
        grid=grid,
        returnMaterial=True
    )
    materialIndex = getMaterialIndex(
        HU_vals=data['HU'],
        CT_data=rawHUvalues
    )
    
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

    # Update direction and position based on sampled angles 
    newDirections, newPositions = updateDirection(
        velocity=directions,
        realAngles=sampleAngles,
        initialPosition=initialPositions
    )

    newEnergies = sampledEnergies
    energyLossStep = initialEnergies - newEnergies
        
    depositEnergy3DStepTraversal(
        initialPositions=initialPositions,
        finalPositions=newPositions,
        energyLossPerStep=energyLossStep,
        initialEnergies=initialEnergies,
        energyDepositedVector=energyGrid,
        fluenceVector=fluenceGrid,
        energyFluenceVector=energyFluenceGrid,
        size=physicalSize,
        bins=gridShape
    )

    if debug:
        print('Initial energies:', initialEnergies)
        print('Material indices:', materialIndex)
        print('Sampled angles:', sampleAngles)
        print('Sampled energies:', sampledEnergies)
        
        print('Energy loss step:', energyLossStep)
        
        print('New directions:\n', newDirections)
        print('New positions:\n', newPositions)
        print('------------------------------------------------\n')
    
    return newPositions, newDirections, newEnergies

# --------------- COMMON FUNCTIONS ---------------
def depositEnergy3DStepTraversal(
    initialPositions : np.ndarray,
    finalPositions : np.ndarray,
    energyLossPerStep : np.ndarray,
    initialEnergies : np.ndarray,
    energyDepositedVector : np.ndarray,
    fluenceVector : np.ndarray,
    energyFluenceVector : np.ndarray,
    size : tuple,
    bins: tuple
):
    '''
    Deposits energy and fluence in a 3D grid locally
    
    Parameters
    ----------
    initialPositions : (N, 3) float32 - initial positions
    finalPositions : (N, 3) float32 - final positions
    energyLossPerStep : (N,) float32 - energy loss per step
    initialEnergies : (N,) float32 - initial energies
    energyDepositedVector : (X, Y, Z) float32 - energy deposited vector
    fluenceVector : (X, Y, Z) float32 - fluence vector
    energyFluenceVector : (X, Y, Z) float32 - energy fluence vector
    size : (3,) float32 - physical size of the world
    bins : (3,) int32 - number of bins (X, Y, Z)
    
    Returns
    -------
    None
    '''
    # Mid‑point & segment length
    midpoints = 0.5 * (initialPositions + finalPositions)
    segmentLength = np.linalg.norm(finalPositions - initialPositions, axis=1)

    # All voxel indices at once
    ix, iy, iz = computeIndices(midpoints, bins, size, grid=None,
                               returnMaterial=False)
    
    depositMultithreaded(
        ix, iy, iz,
        energyLossPerStep,
        segmentLength,
        initialEnergies,
        energyDepositedVector,
        fluenceVector,
        energyFluenceVector
    )

# --------------- COMMON FUNCTIONS ---------------
@numba.jit(nopython=True, fastmath=True)
def depositMultithreaded(
    ix, iy, iz,
    energyLossPerStep,
    segmentLength,
    initialEnergies,
    energyDepositedVector,
    fluenceVector,
    energyFluenceVector
):
    """
    Core function for multithreaded energy deposition using Numba.
    """
    nParticles = ix.shape[0]

    # Use numba.prange to parallelize the loop
    for i in prange(nParticles):
        # A simple check to ensure indices are within bounds
        # This is a safe guard, but your computeIndices should handle it
        if 0 <= ix[i] < energyDepositedVector.shape[0] and \
           0 <= iy[i] < energyDepositedVector.shape[1] and \
           0 <= iz[i] < energyDepositedVector.shape[2]:

            energyDepositedVector[ix[i], iy[i], iz[i]] += energyLossPerStep[i]
            fluenceVector[ix[i], iy[i], iz[i]] += segmentLength[i]
            
            eMid = initialEnergies[i] - 0.25 * energyLossPerStep[i]
            energyFluenceVector[ix[i], iy[i], iz[i]] += eMid * segmentLength[i]
            

# --------------- COMMON FUNCTIONS ---------------
def simulateBatchParticlesVectorized(
    batchSize, data, gridMap, initialEnergy,
    bigVoxelSize, energyDepositedVector, fluenceVector, 
    energyFluenceVector, cdfs, debug, binEdges, 
    method='transformation',
    angleStep=None, energyStep=None
):
    # Convert to numpy arrays
    
    bigVoxelSize = np.array(bigVoxelSize)
    gridShape = np.array(gridMap.shape)
    
    # Random -10 mm to +10 mm in X and Y axis
    initialZ = -150.  # Initial position in mm
    xyRange = 0.  # Half-width in mm (so total 50 mm x 50 mm field)
    
    # Generate uniform random positions in X and Y 
    x = np.random.uniform(-xyRange, xyRange, batchSize)
    y = np.random.uniform(-xyRange, xyRange, batchSize)
    z = np.full(batchSize, initialZ)
    position = np.stack([x, y, z], axis=1)
    
    energy = np.full(batchSize, initialEnergy)
    velocity = np.tile([0.0, 0.0, 1.0], (batchSize, 1))
    active = np.ones(batchSize, dtype=bool)

    if debug:
        print(f'------------------------------------------------')
        print(f'Initial Energy: {initialEnergy} MeV')
        print(f'Initial Position: {position}')
        print(f'------------------------------------------------')


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
            debug=debug,
            binEdges=binEdges,
            gridShape=gridShape,
            physicalSize=bigVoxelSize,
            method=method,
            angleStep=angleStep,
            energyStep=energyStep
        )
        
        # Check if particles are still active
        withinBounds = np.all(
            (position[active] >= -bigVoxelSize) & (position[active] < bigVoxelSize),
            axis=1
        )
        realEnergiesValid = energy[active] > 0
        active[active] = realEnergiesValid & withinBounds

       
# --------------- COMMON FUNCTIONS ---------------
def updateDirection(velocity, realAngles, initialPosition):
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
    
    vz = newVelocity[:, 2:3]
    step = newVelocity / vz  # ensures dz = 1 mm
    newPosition = initialPosition + step

    return newVelocity, newPosition

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
def simulateBatchParticlesWorker(args):
    method = args[24] 
    batchID = args[25]
    
    # Common extraction for all methods
    (
        shm_name, shape, dtype_str,
        shm_cdfs_name, cdfs_shape, cdfs_dtype_str,
        shm_energy_deposited_name, energy_deposited_shape, energy_deposited_dtype_str,
        shm_fluence_name, fluence_shape, fluence_dtype_str,
        shm_energy_fluence_name, energy_fluence_shape, energy_fluence_dtype_str,
        batchSize, materials, energies, minEnergyByMaterial,
        gridMaterial, initialEnergy,
        bigVoxelSize, seed, debug, _, _  # method and batchID are already extracted
    ) = args[:26]

    # Default to None (for both methods)
    binEdges = angleStep = energyStep = None
    shm_bin_edges_name = bin_edges_shape = bin_edges_dtype_str = None

    # Handle method-specific arguments
    if method == 'normalization':
        (
            shm_bin_edges_name, bin_edges_shape, bin_edges_dtype_str
        ) = args[26:]

    elif method == 'transformation':
        (
            binEdges, angleStep, energyStep
        ) = args[26:]

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
        'HU': materials,
        'energies': energies,
        'minEnergyByMaterial': minEnergyByMaterial
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
        debug=debug,
        binEdges=binEdges,
        method=method,
        angleStep=angleStep,
        energyStep=energyStep
    )

    # Safely merge local vectors into shared memory (atomic if required)
    np.add(shared_energyDeposited, localEnergyDeposited, out=shared_energyDeposited)
    np.add(shared_fluence, localFluence, out=shared_fluence)
    np.add(shared_energyFluence, localEnergyFluence, out=shared_energyFluence)
    
    # Optional: logging
    total_energy = np.sum(localEnergyDeposited)
    nonzero_voxels = np.sum(localEnergyDeposited > 0)
    with open("logs.txt", "a") as log_file:
        log_file.write(f"Batch {batchID:04d} | Deposited Energy: {total_energy:.2f} MeV | Nonzero Deposits: {nonzero_voxels}\n")

    return result

# --------------- COMMON FUNCTIONS ---------------
def runMultiprocessedBatchedSim(
    totalSamples, batchSize, numWorkers,
    shm_prob_table, prob_table_shape, prob_table_dtype,
    shm_cdfs, cdfs_shape, cdfs_dtype,
    shm_energy_deposited, energy_deposited_shape, energy_deposited_dtype,
    shm_fluence, fluence_shape, fluence_dtype,
    shm_energy_fluence, energy_fluence_shape, energy_fluence_dtype,
    data, gridMaterial, initialEnergy, 
    bigVoxelSize, debug, method = 'transformation',
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
            min(batchSize, totalSamples - i * batchSize),
            data['HU'], data['energies'], data['minEnergyByMaterial'],
            gridMaterial, initialEnergy,
            bigVoxelSize, currentSeed, debug,
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
        
# --------------- COMMON FUNCTIONS ---------------
def loadCTVolume(
    dicomDir: str,
    sortBy: str = 'z',           # 'z' (default) or 'filename'
    dtype: np.dtype = np.int32
): 
    """
    Read all DICOM files in *dicomDi* and return a 3‑D numpy array of HU values.

    Parameters
    ---------+
    
    dicomDi : str
        Path to the folder that contains the DICOM series.
    sort_by : str, optional
        'z'   – sort by ImagePositionPatient[2] (recommended for CT).
        'filename' – lexicographic order of the file names.
    dtype : np.dtype, optional
        Desired dtype of the returned array (default int16).

    Returns
    -------
    volume : np.ndarray
        3‑D array with shape (Rows, Columns, Slices).  HUs are signed integers.
    """
    # Gather all DICOM file paths
    dicomPaths = [os.path.join(dicomDir, f)
                   for f in os.listdir(dicomDir)
                   if f.lower().endswith('.dcm')]

    if not dicomPaths:
        raise FileNotFoundError(f"No DICOM files found in {dicomDir}")

    # Load a few slices to discover image geometry
    sampleds = pydicom.dcmread(dicomPaths[0])
    rows, cols = int(sampleds.Rows), int(sampleds.Columns)

    # Build a list of (path, z‑index) tuples
    slices = []
    for path in dicomPaths:
        ds = pydicom.dcmread(path, stop_before_pixels=True)  # just read meta
        # Use the z‑coordinate if available; otherwise fallback to file name
        if sortBy == 'z':
            # Some series use ImagePositionPatient; others use InstanceNumber.
            z = float(ds.get('ImagePositionPatient', [0, 0, 0])[2])
        else:
            z = float(os.path.basename(path).split('.')[0])  # crude fallback
        slices.append((path, z))

    # Sort slices
    slices.sort(key=lambda x: x[1])

    # Allocate the volume array
    nSlices = len(slices)
    volume = np.empty((rows, cols, nSlices), dtype=dtype)
    
    # Loop over the sorted slices, read pixel data, apply rescaling
    for idx, (path, _) in enumerate(slices):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)  # keep float until rescale

        # Rescale to HUs if the series includes it
        intercept = float(ds.get('RescaleIntercept', 0.0))
        slope     = float(ds.get('RescaleSlope', 1.0))
        img = img * slope + intercept

        # Clip to signed 16‑bit range (optional)
        img = np.clip(img, -32768, 32767)

        # Store in the volume
        volume[:, :, idx] = img.astype(dtype)

    return volume
# --------------- MAIN FUNCTION ----------------
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--transformation', action='store_true', help="Use transformation method")
    group.add_argument('--normalization', action='store_true', help="Use normalization method")
    parser.add_argument('--debug', action='store_true', help="Use debug mode")
    parser.add_argument('--denoise', action='store_true', help="Use denoising dataset")
    args = parser.parse_args()
    
    method = 'transformation' if args.transformation else 'normalization'
    denoised = True if args.denoise else False

    debug = True if args.debug else False
    print(f'[INFO] Using {method} method.')
    
    # Start timing
    startTime = time.time()
    print(f'[INFO] Starting simulation timer ...')
    
    # Shared settings
    samplingN = 100_000
    initialEnergy = 200.  # MeV
    bigVoxelSize = np.array((100., 100., 150.), dtype=np.float32) # in mm
    voxelShapeBins = np.array((50, 50, 300), dtype=np.int32)
    voxelSize = 2 * bigVoxelSize / voxelShapeBins # in mm 
    
    print(f'[INFO] Using voxel size: {bigVoxelSize} mm centered at (0, 0, 0)')
    print(f'[INFO] Using voxel shape: {voxelShapeBins}')
    print(f'[INFO] Using voxel spacing: {voxelSize} mm')

    angleRange = (0, 70)
    energyRange = (-0.6, 0)
    angleBins = 100
    energyBins = 100
        
    # Load DICOM CT
    if debug:
        print('[INFO] Loading CT volume...')
        
    gridMaterial = np.zeros((voxelShapeBins[0], voxelShapeBins[1], voxelShapeBins[2]), dtype=np.int32)
    dicomPath = './dicom_data'
    if not os.path.exists(dicomPath):
        raise FileNotFoundError(f"CT directory not found at {dicomPath}. Please ensure the CT is generated first. Loading default CT with zeros.")
    else:
        gridMaterial = loadCTVolume(dicomPath, sortBy='z')
    if debug:
        print('[INFO] CT volume loaded successfully. Phantom shape:', gridMaterial.shape)
        
    file_paths = {
        'denoised': {
            'transformation': {
                'npz': './denoised_histograms_transformation.npz',
                'npy': './NumpyTrans/',
                'csv': './CSVTrans/'
            },
            'normalization': {
                'npz': './denoised_histograms_normalization.npz',
                'npy': './NumpyNorm/',
                'csv': './CSVNorm/'
            }
        },
        'raw': {  # Using a more descriptive name for the non-denoised files
            'transformation': {
                'npz': './DenoisingDataTransSheetSliced.npz',
                'npy': './NumpyTrans/',
                'csv': './CSVTrans/'
            },
            'normalization': {
                'npz': './DenoisingDataNormSheetSliced.npz',
                'npy': './NumpyNorm/',
                'csv': './CSVNorm/'
            }
        }
    }

    # Select the correct path configuration based on the 'denoised' and 'method' flags.
    data_type = 'denoised' if denoised else 'raw'
    config = file_paths[data_type][method]

    # Assign the paths to variables for clarity
    npzPath = config['npz']
    npyPath = config['npy']
    csvPath = config['csv']

    # Make sure the directories exist. `Path` from `pathlib` is the modern standard.
    # It's more robust and works on all operating systems.
    Path(npyPath).mkdir(parents=True, exist_ok=True)
    Path(csvPath).mkdir(parents=True, exist_ok=True)

    # You can now use npzPath, npyPath, and csvPath in the rest of your code.
    print(f"[INFO] Using NPZ path: {npzPath}")
    print(f"[INFO] Using NPY directory: {npyPath}")
    print(f"[INFO] Using CSV directory: {csvPath}")

    # --- Load table ---
    rawData = np.load(npzPath, mmap_mode='r', allow_pickle=True)
    
    print(f'\n[INFO] Loaded {npzPath}...')
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[INFO] Memory usage after loading npz file: {mem_info.rss / 1024 / 1024:.2f} MB")
    
    probTable = rawData['probTable']
    materials = rawData['HU']
    energies = rawData['energies']

    # Share data fields
    data = {
        'probTable': probTable,
        'HU': materials,
        'energies': energies,
    }
    
    del probTable, materials, energies
    
    if method == 'normalization':
        # Load extra arrays only used in normalization
        data['thetaMax'] = rawData['thetaMax'][0]
        data['thetaMin'] = rawData['thetaMin'][0]
        data['energyMin'] = rawData['energyMin'][0]
        data['energyMax'] = rawData['energyMax'][0]
    
    rawData.close()
    del rawData
    
    # Print keys in the dictionary for info
    if debug:
        print(f'[INFO] Keys and values in the dictionary:')
        for key in data.keys():
            print(f"\tKey: {key}, Shape: {data[key].shape}, Type: {type(data[key])}")

    binEdges = None
    minEnergyByMaterial = None
    
    if debug:
        print('[INFO] Materials(HU values) and energies available:')
        print(f'\tMaterials: {data["HU"]}')
        print(f'\tEnergies: {data["energies"]}')
            
    # Build CDFs 
    if debug: 
        print(f'[INFO] Building CDFs...')
    if method == 'normalization':
        cdfs, binEdges, minEnergyByMaterial = buildCdfsAndCompactBins(data=data)      
        # Remove unnecessary keys from data to free memory
        for key in ['thetaMax', 'thetaMin', 'energyMin', 'energyMax']:
            del data[key]
    else:
        cdfs, minEnergyByMaterial = buildCdfsFromProbTable(data['probTable'], data['energies'])
    
    if debug: 
        print('[INFO] CDFs built.')
        print('[INFO] CDFs shape:', cdfs.shape)
        if method == 'normalization':
            print('[INFO] Bin edges shape:', binEdges.shape)

    if debug:
        print('[INFO] Material HU values -> minimum energy for each material:')
        print('[INFO] Shape:', minEnergyByMaterial.shape)
        for mIndex in range(len(data['HU'])):
            print(f'\t{data["HU"][mIndex]} -> {minEnergyByMaterial[mIndex]:.2f} MeV')
        
    # Add minEnergyByMaterial to data
    data['minEnergyByMaterial'] = minEnergyByMaterial

    # Shared memory
    energyDeposited = np.zeros(voxelShapeBins, dtype=np.float32)
    shm_prob_table, prob_table_shape, prob_table_dtype = createSharedMemory(data['probTable'])
    shm_cdfs, cdfs_shape, cdfs_dtype = createSharedMemory(cdfs)
    shm_energy_deposited, energyDeposited_shape, energyDeposited_dtype = createSharedMemory(energyDeposited)
    
    # Shared memory for fluence and energyFluence
    fluence = np.zeros(voxelShapeBins, dtype=np.float32)
    energyFluence = np.zeros(voxelShapeBins, dtype=np.float32)
    shm_fluence, fluence_shape, fluence_dtype = createSharedMemory(fluence)
    shm_energy_fluence, energyFluence_shape, energyFluence_dtype = createSharedMemory(energyFluence)

    if debug:
        print(f'[INFO] Scorers available:')
        print(f'\t.Energy Deposited: {energyDeposited.shape}')
        print(f'\t.Fluence: {fluence.shape}')
        print(f'\t.Energy Fluence: {energyFluence.shape}')
    
    # Multiprocessing
    batchSize = 10000
    numWorkers = cpu_count()
    if debug:
        print('[INFO] Multiprocessing settings:')
        print(f'\tSampling N: {samplingN}')
        print(f'\tInitial energy: {initialEnergy}')
        print(f'\tBatch size: {batchSize}')
        print(f'\tNumber of workers: {numWorkers}')
    
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
        
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"\n[INFO] Memory Usage before running simulation: {mem_info.rss / 1024 / 1024:.2f} MB")
    
    # Run simulation
    print(f"[INFO] Running simulation using '{method}' method.")

    runMultiprocessedBatchedSim(
        samplingN, batchSize, numWorkers,
        shm_prob_table, prob_table_shape, prob_table_dtype,
        shm_cdfs, cdfs_shape, cdfs_dtype,
        shm_energy_deposited, energyDeposited_shape, energyDeposited_dtype,
        shm_fluence, fluence_shape, fluence_dtype,
        shm_energy_fluence, energyFluence_shape, energyFluence_dtype,
        data, gridMaterial, initialEnergy,
        bigVoxelSize, debug, method,
        **kwargs
    )
                    
    energyVector3D = np.ndarray(energyDeposited.shape, dtype=energyDeposited.dtype, buffer=shm_energy_deposited.buf).copy()
    np.save(Path(npyPath) / f'energyDeposited{method}.npy', energyVector3D)
    
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
    print(f"[INFO] Total energy: {totalEnergy:.6f} MeV")
    
    fileforEnergyDeposit = f"{csvPath}EnergyAtBoxByBinsMySimulation_{method}.csv"
    if debug:
        print(f"[INFO] Writing to {fileforEnergyDeposit}...")
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
                    
    if debug:
        print(f"[INFO] Energy deposited written to {fileforEnergyDeposit}.")
                                    
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
    
    if method == 'normalization':
        shm_bin_edges.unlink()
        shm_bin_edges.close()
        
    endTime = time.time()
    print(f"[INFO] Simulation time: {endTime - startTime:.12f} seconds")
    print()