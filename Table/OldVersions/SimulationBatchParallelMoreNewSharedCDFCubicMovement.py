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
import seaborn as sns
import pandas as pd
from typing import Tuple
from pathlib import Path
from scipy.stats import qmc

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
    
    # Material-specific minimum valid energy per material   
    minEnergyByMaterial = {
        'G4_WATER': 9.0,
        'G4_LUNG_ICRP': 9.5,
        'G4_BONE_CORTICAL_ICRP': 12.0,
        'G4_TISSUE_SOFT_ICRP': 9.5
    }
    materialMinEnergy = minEnergyByMaterial.get(material, np.min(availableEnergies))

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
    mask = energies < materialMinEnergy
    realAngles[mask] = 0
    realEnergies[mask] = 0

    return realAngles, realEnergies

def sampleFromCDFVectorizedTransformationCubic(
    data: dict, 
    material: str, 
    energies: np.ndarray, 
    materialToIndex: dict, 
    cdfs: np.ndarray,
    binEdges: np.ndarray, 
    angleStep: float, 
    energyStep: float
):
    materialIdx = materialToIndex[material]
    availableEnergies = data['energies']
    angleBins, energyBins = data['probTable'].shape[2:]

    # Material-specific minimum energy
    minEnergyByMaterial = {
        'G4_WATER': 9.0,
        'G4_LUNG_ICRP': 9.5,
        'G4_BONE_CORTICAL_ICRP': 12.0,
        'G4_TISSUE_SOFT_ICRP': 9.5
    }
    materialMinEnergy = minEnergyByMaterial.get(material, np.min(availableEnergies))

    minEnergy = np.min(availableEnergies)
    maxEnergy = np.max(availableEnergies)
    roundedEnergies = np.clip(np.round(energies, 1), minEnergy, maxEnergy)

    # Reverse for interpolation
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

        uniqueEnergyIndices = np.unique(indices)
        for energyIdx in uniqueEnergyIndices:
            idxs = np.where(indices == energyIdx)[0].astype(np.int32)

            angleEdges = binEdges[0, :angleBins + 1]
            energyEdges = binEdges[1, :energyBins + 1]

            if angleEdges[0] == angleEdges[-1] or energyEdges[0] == energyEdges[-1]:
                out[label + '_angles'][idxs] = 0.0
                out[label + '_energies'][idxs] = 0.0
                continue

            sampleCDFForEnergyGroup(
                materialIdx, energyIdx, idxs, rand[idxs],
                cdfs, angleBins, energyBins, angleEdges, energyEdges,
                angleStep, energyStep, out[label + '_angles'], out[label + '_energies']
            )

    interpolateMask = energies >= materialMinEnergy

    # Catmull-Rom interpolation in transformed space
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

    # Transform back to real space
    realAngles, realEnergies = reverseVariableChangeTransform(energies, sampledAngles, sampledEnergies)

    # Mask below-threshold input energies
    realAngles[~interpolateMask] = 0
    realEnergies[~interpolateMask] = 0

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
# def pchip_interpolate_1d(f_1, f0, f1, f2, t):
#     """
#     Apply 1D PCHIP interpolation using 4 neighboring values.
#     Each set (f_1, f0, f1, f2) is interpolated at location t ∈ [0, 1].
#     """
#     # Create x-grid centered on f0-f1 region
#     x = np.array([-1, 0, 1, 2], dtype=np.float32)
#     y = np.stack([f_1, f0, f1, f2], axis=-1)  # shape: (N, 4)
    
#     # Perform PCHIP interpolation per row
#     interpolated = np.zeros_like(f0)
#     for i in range(f0.shape[0]):
#         interpolator = PchipInterpolator(x, y[i], extrapolate=True)
#         interpolated[i] = interpolator(t[i])
#     return interpolated

# def pchip_interpolate_1d(f_1, f0, f1, f2, t):
#     """
#     Vectorized PCHIP interpolation using scipy's pchip_interpolate.
#     Inputs:
#         - f_1, f0, f1, f2: arrays of shape (N,)
#         - t: interpolation location array of shape (N,), values in [0, 1]
#     Output:
#         - interpolated values at t ∈ [0, 1]
#     """
#     x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)  # fixed x for all
#     y_all = np.stack([f_1, f0, f1, f2], axis=1)            # shape (N, 4)
#     t_query = t + 0.0  # move t from [0, 1] to relative scale on x

#     # Vectorized interpolation via list comprehension (faster than loop with PchipInterpolator)
#     interpolated = np.array([
#         pchip_interpolate(x, y_row, t_val)
#         for y_row, t_val in zip(y_all, t_query)
#     ], dtype=np.float32)

#     return interpolated

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
    


# def sampleFromCDFVectorizedNormalizationCubic(
#     data: dict, 
#     material: str, 
#     energies: np.ndarray, 
#     materialToIndex: dict,
#     cdfs: np.ndarray, 
#     binEdges: np.ndarray
# ) -> Tuple[np.ndarray, np.ndarray]:

#     materialIdx = materialToIndex[material]
#     availableEnergies = data['energies']
#     angleBins, energyBins = data['probTable'].shape[2:]
    
#     # Material-specific minimum valid energy per material   
#     minEnergyByMaterial = {
#         'G4_WATER': 9.3,
#         'G4_LUNG_ICRP': 9.8,
#         'G4_BONE_CORTICAL_ICRP': 12.0,
#         'G4_TISSUE_SOFT_ICRP': 9.8
#     }
#     materialMinEnergy = minEnergyByMaterial.get(material, np.min(availableEnergies))

#     minEnergy = np.min(availableEnergies)
#     maxEnergy = np.max(availableEnergies)
#     # roundedEnergies = np.clip(np.round(energies, 1), minEnergy, maxEnergy)
#     roundedEnergies = np.clip(energies, minEnergy, maxEnergy)

#     # Interpolation setup
#     reversedEnergies = availableEnergies[::-1]
#     insertPos = np.searchsorted(reversedEnergies, roundedEnergies, side='left')
#     insertPos = np.clip(insertPos, 1, len(reversedEnergies) - 2)

#     base = len(availableEnergies) - 1
#     i1 = base - insertPos
#     i0 = np.clip(i1 - 1, 0, base)
#     i2 = np.clip(i1 + 1, 0, base)
#     i_1 = np.clip(i1 - 2, 0, base)

#     e0 = availableEnergies[i0]
#     e1 = availableEnergies[i1]
    
#     # print('Lower energy: ', lowerEnergy)
#     # print('Upper energy: ', upperEnergy)

#     # Interpolation weights, only applied where energy >= minEnergy
#     weights = np.where(
#         e1 != e0,
#         (roundedEnergies - e0) / (e1 - e0),
#         0.0
#     ).astype(np.float32)

#     rand = np.random.random(size=energies.shape).astype(np.float32)

#     out = {}
#     for label, indices in zip(
#         ['_1', '0', '1', '2'], [i_1, i0, i1, i2]
#     ):
#         out[label + '_angles'] = np.zeros_like(energies, dtype=np.float32)
#         out[label + '_energies'] = np.zeros_like(energies, dtype=np.float32)

#         uniqueEnergyIndices = np.unique(indices)
#         for energyIdx in uniqueEnergyIndices:
#             idxs = np.where(indices == energyIdx)[0].astype(np.int32)

#             angleEdges = binEdges[materialIdx, energyIdx, 0, :angleBins + 1]
#             energyEdges = binEdges[materialIdx, energyIdx, 1, :energyBins + 1]

#             if angleEdges[0] == angleEdges[-1] or energyEdges[0] == energyEdges[-1]:
#                 out[label + '_angles'][idxs] = 0.0
#                 out[label + '_energies'][idxs] = 0.0
#                 continue

#             angleStep = angleEdges[1] - angleEdges[0]
#             energyStep = energyEdges[1] - energyEdges[0]

#             sampleCDFForEnergyGroup(
#                 materialIdx, energyIdx, idxs, rand[idxs],
#                 cdfs, angleBins, energyBins, angleEdges, energyEdges,
#                 angleStep, energyStep, out[label + '_angles'], out[label + '_energies']
#             )

#     aboveMinEnergy = energies >= materialMinEnergy
#     braggPeak = (roundedEnergies < 40) | (roundedEnergies >= materialMinEnergy)
#     interpolateMask = aboveMinEnergy & braggPeak

#     # Use Catmull-Rom interpolation where allowed, else use direct samples from i1
#     sampledAngles = np.where(
#         interpolateMask,
#         catMullRomInterpolation(
#             out['_1_angles'], out['0_angles'], out['1_angles'], out['2_angles'], weights
#         ),
#         out['1_angles']
#     )
#     sampledEnergies = np.where(
#         interpolateMask,
#         catMullRomInterpolation(
#             out['_1_energies'], out['0_energies'], out['1_energies'], out['2_energies'], weights
#         ),
#         out['1_energies']
#     )
    
#     sampledEnergies[~aboveMinEnergy] = 0.0
#     sampledAngles[~aboveMinEnergy] = 0.0

#     return sampledAngles, sampledEnergies
    
# --------------- NORMALIZATION VARIABLE -------------
def sampleFromCDFVectorizedNormalizationCubic(
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
    
    # Material-specific minimum valid energy per material   
    minEnergyByMaterial = {
        'G4_WATER': 9.0,
        'G4_LUNG_ICRP': 9.5,
        'G4_BONE_CORTICAL_ICRP': 12.0,
        'G4_TISSUE_SOFT_ICRP': 9.5
    }
    materialMinEnergy = minEnergyByMaterial.get(material, np.min(availableEnergies))

    minEnergy = np.min(availableEnergies)
    maxEnergy = np.max(availableEnergies)
    # roundedEnergies = np.clip(np.round(energies, 1), minEnergy, maxEnergy)
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
    
    # print('Lower energy: ', lowerEnergy)
    # print('Upper energy: ', upperEnergy)

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

        uniqueEnergyIndices = np.unique(indices)
        for energyIdx in uniqueEnergyIndices:
            idxs = np.where(indices == energyIdx)[0].astype(np.int32)

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

    interpolateMask = energies >= materialMinEnergy
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

# def sampleFromCDFVectorizedNormalizationCubic(
#     data: dict, 
#     material: str, 
#     energies: np.ndarray, 
#     materialToIndex: dict,
#     cdfs: np.ndarray, 
#     binEdges: np.ndarray
# ) -> Tuple[np.ndarray, np.ndarray]:

#     materialIdx = materialToIndex[material]
#     availableEnergies = data['energies']
#     angleBins, energyBins = data['probTable'].shape[2:]
    
#     # Material-specific minimum valid energy per material   
#     minEnergyByMaterial = {
#         'G4_WATER': 9.3,
#         'G4_LUNG_ICRP': 9.8,
#         'G4_BONE_CORTICAL_ICRP': 12.0,
#         'G4_TISSUE_SOFT_ICRP': 9.8
#     }
#     materialMinEnergy = minEnergyByMaterial.get(material, np.min(availableEnergies))

#     minEnergy = np.min(availableEnergies)
#     maxEnergy = np.max(availableEnergies)
#     roundedEnergies = np.clip(energies, minEnergy, maxEnergy)

#     reversedEnergies = availableEnergies[::-1]
#     insertPos = np.searchsorted(reversedEnergies, roundedEnergies, side='left')
#     insertPos = np.clip(insertPos, 1, len(reversedEnergies) - 2)

#     base = len(availableEnergies) - 1
#     i1 = base - insertPos
#     i0 = np.clip(i1 - 1, 0, base)
#     i2 = np.clip(i1 + 1, 0, base)
#     i_1 = np.clip(i1 - 2, 0, base)

#     e0 = availableEnergies[i0]
#     e1 = availableEnergies[i1]

#     # Interpolation weights only where e1 != e0
#     weights = np.where(
#         e1 != e0,
#         (roundedEnergies - e0) / (e1 - e0),
#         0.0
#     ).astype(np.float32)

#     rand = np.random.random(size=energies.shape).astype(np.float32)

#     out = {}
#     for label, indices in zip(
#         ['_1', '0', '1', '2'], [i_1, i0, i1, i2]
#     ):
#         out[label + '_angles'] = np.zeros_like(energies, dtype=np.float32)
#         out[label + '_energies'] = np.zeros_like(energies, dtype=np.float32)

#         uniqueEnergyIndices = np.unique(indices)
#         for energyIdx in uniqueEnergyIndices:
#             idxs = np.where(indices == energyIdx)[0].astype(np.int32)

#             angleEdges = binEdges[materialIdx, energyIdx, 0, :angleBins + 1]
#             energyEdges = binEdges[materialIdx, energyIdx, 1, :energyBins + 1]

#             if angleEdges[0] == angleEdges[-1] or energyEdges[0] == energyEdges[-1]:
#                 out[label + '_angles'][idxs] = 0.0
#                 out[label + '_energies'][idxs] = 0.0
#                 continue

#             angleStep = angleEdges[1] - angleEdges[0]
#             energyStep = energyEdges[1] - energyEdges[0]

#             sampleCDFForEnergyGroup(
#                 materialIdx, energyIdx, idxs, rand[idxs],
#                 cdfs, angleBins, energyBins, angleEdges, energyEdges,
#                 angleStep, energyStep, out[label + '_angles'], out[label + '_energies']
#             )

#     interpolateMask = energies >= materialMinEnergy

#     # Detect exact matches (within some tolerance) to energies in availableEnergies[i1]
#     # Use e1 (same shape as energies) from above
#     tol = 1e-6
#     exactMatchMask = np.abs(roundedEnergies - e1) < tol

#     # Initialize outputs
#     sampledAngles = np.zeros_like(energies, dtype=np.float32)
#     sampledEnergies = np.zeros_like(energies, dtype=np.float32)

#     # For exact matches, directly take samples from i1 bins (out['1_angles'], out['1_energies'])
#     sampledAngles[exactMatchMask] = out['1_angles'][exactMatchMask]
#     sampledEnergies[exactMatchMask] = out['1_energies'][exactMatchMask]

#     # For others (where interpolateMask is True and not exact match), do interpolation
#     interpIndices = interpolateMask & (~exactMatchMask)

#     sampledAngles[interpIndices] = catMullRomInterpolation(
#         out['_1_angles'][interpIndices], 
#         out['0_angles'][interpIndices], 
#         out['1_angles'][interpIndices], 
#         out['2_angles'][interpIndices], 
#         weights[interpIndices]
#     )
#     sampledEnergies[interpIndices] = catMullRomInterpolation(
#         out['_1_energies'][interpIndices], 
#         out['0_energies'][interpIndices], 
#         out['1_energies'][interpIndices], 
#         out['2_energies'][interpIndices], 
#         weights[interpIndices]
#     )

#     # For energies below minEnergy, sampled outputs remain 0

#     return sampledAngles, sampledEnergies

# --------------- NORMALIZATION VARIABLE -------------
@numba.jit(nopython=True, inline = 'always')
def applyNormalizationScaling(energyActive, energyLoss, dt, energyThreshold=20.0, kMin=0.7, kMax=1.0):
    n = energyActive.shape[0]
    dtScaled = np.full_like(energyActive, dt)
    energyLossScaled = energyLoss.copy()

    for i in range(n):
        if energyActive[i] < energyThreshold:
            k = kMin + (kMax - kMin) * (energyActive[i] / energyThreshold)
            dtScaled[i] *= k
            energyLossScaled[i] *= k
        else:
            # Keep original dt and energyLoss for energies >= threshold
            dtScaled[i] = dt
            energyLossScaled[i] = energyLoss[i]

    return dtScaled, energyLossScaled

# --------------- NORMALIZATION VARIABLE -------------
@numba.jit(nopython=True, inline='always')
def applyNormalizationScalingStochastic(energyActive, energyLoss, dt, noise, 
                                        energyThreshold=20.0, kMin=0.7, kMax=1.0, alpha=0.1):
    n = energyActive.shape[0]
    dtScaled = np.full_like(energyActive, dt)
    energyLossScaled = energyLoss.copy()

    for i in range(n):
        if energyActive[i] < energyThreshold:
            mean_k = kMin + (kMax - kMin) * (energyActive[i] / energyThreshold)
            std_k = alpha * mean_k
            k = mean_k + std_k * noise[i]  # z ~ N(0, 1)
            dtScaled[i] *= k
            energyLossScaled[i] *= k
        else:
            # Keep original dt and energyLoss for energies >= threshold
            dtScaled[i] = dt
            energyLossScaled[i] = energyLoss[i]

    return dtScaled, energyLossScaled


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
        
#     return energyDepositedVector# --------------- COMMON FUNCTIONS ---------------
@numba.jit(nopython=True)
def calculateEnergyDepositBinBatch(
    positions: np.ndarray,
    physicalSize: tuple,
    energyLosses: np.ndarray,
    energyDepositedVector: np.ndarray,
    voxelShapeBins: tuple
):
    n = positions.shape[0]
    sizeX, sizeY, sizeZ = physicalSize
    binsX, binsY, binsZ = voxelShapeBins

    # Define min corner of the volume
    minX = -sizeX
    minY = -sizeY
    minZ = -sizeZ

    # Define bin width in each direction
    binWidthX = (2 * sizeX) / binsX
    binWidthY = (2 * sizeY) / binsY
    binWidthZ = (2 * sizeZ) / binsZ

    for i in prange(n):
        # Compute index by shifting position and dividing by bin width
        ix = int((positions[i, 0] - minX) / binWidthX)
        iy = int((positions[i, 1] - minY) / binWidthY)
        iz = int((positions[i, 2] - minZ) / binWidthZ)

        # Clamp to valid range
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
        # print(f"Energy deposited in voxel bin {ix}, {iy}, {iz} : {energyLosses[i]}")

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
        return sampleFromCDFVectorizedNormalizationCubic(
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
    binEdges, method='transformation',
    angleStep=None, energyStep=None
):
    energy = np.full(batchSize, initialEnergy)
    # initialZ = -bigVoxelSize[2] + 1.2 * np.random.rand(batchSize)  
    # initialZ = -bigVoxelSize[2] + np.random.uniform(-0.5, 0.5, size=batchSize) #   Uniform random offset of ±0.5 mm around the voxel center to distribute starts across the full voxel.
    # initialZ = -bigVoxelSize[2] + np.clip(np.random.normal(0.0, 0.3, size=batchSize), -0.5, 0.5) # Normal (Gaussian) jitter centered at voxel edge, clipped to ±0.5 mm to prevent large deviations.
             
    # offsets = np.linspace(-0.5, 0.5, batchSize)
    # np.random.shuffle(offsets)
    # initialZ = -bigVoxelSize[2] + offsets   #  Evenly spaced offsets over ±0.5 mm, randomly shuffled to ensure well-distributed but non-overlapping starts.
    
    # sampler = qmc.Halton(d=1, scramble=True)
    # initialZ = -bigVoxelSize[2] + qmc.scale(sampler.random(batchSize), 0.0, 1.0).flatten()  # Uses a Halton low-discrepancy sequence (scrambled) for quasi-random sampling from 0 to 1 mm to evenly fill space without clustering.
        
    # position = np.column_stack((np.zeros(batchSize), np.zeros(batchSize), initialZ))
    position = np.tile([0.0, 0.0, -bigVoxelSize[2]], (batchSize, 1))
    
    velocity = np.tile([0.0, 0.0, 1.0], (batchSize, 1))  # Initial velocity along Z
    active = np.ones(batchSize, dtype=bool)
    
    # zStep = (2 * bigVoxelSize[2] / energyDepositedVector.shape[2])
    zStep = 1  # Fixed step of 1 mm
    # zStepPattern = np.array([1.25, 0.75])  # mm steps
    # stepIndex = 0  # start with 1.0 mm
    
    while np.any(active):
        energyActive = energy[active]

        # Sample angles and energies
        realAngles, realEnergies = sampleFromCDF(
            data=data, material=material, energies=energyActive,
            materialToIndex=materialToIndex, cdfs=cdfs, binEdges=binEdges,
            method=method, angleStep=angleStep, energyStep=energyStep
        )
        energyLossPerStep = energyActive - realEnergies
        energy[active] = realEnergies
        
        # print(f'Energy: {energy[active]}')
        # print(f'Position: {position[active]}')

        calculateEnergyDepositBinBatch(
            position[active], bigVoxelSize, energyLossPerStep,
            energyDepositedVector, energyDepositedVector.shape
        )
                
        # Alternate zStep each iteration
        # zStep = 1.5 + np.random.rand(np.sum(active)) # Steps range from 0.5 to 1.5 mm, uniformly random — avoids syncing with voxel edges.
        # zStep = 1.0 + np.random.uniform(-0.25, 0.25, size=np.sum(active)) # Steps vary slightly around 1.0 mm (±0.25) — keeps movement near nominal while breaking alignment.
        # zStep = np.clip(np.random.normal(1.0, 0.25, size=np.sum(active)), 0.5, 1.5) # Normally distributed steps around 1.0 mm, clipped to avoid extremes 
        
        # zStep = np.random.choice([0.5, 1.0, 1.5], size=np.sum(active))  # Randomly chooses between fixed discrete step sizes — mixes regular spacing with randomness.
        # zStep = 1.0 + (np.random.rand(np.sum(active)) - 0.5) * 0.5  # Steps vary in a narrow band (0.75 to 1.25 mm) — controlled jitter centered at 1.0 mm.

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
    # displacement = v_rot * step_length[:, None]

    new_position = position + displacement
    new_velocity = v_rot

    return new_position, new_velocity
        
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
    
    baseSeed = 76743563578
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
    material = 'G4_BONE_CORTICAL_ICRP'
    initialEnergy = 200.  # MeV
    bigVoxelSize = np.array((100., 100., 150.), dtype=np.float64)
    voxelShapeBins = (50, 50, 200)
    
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