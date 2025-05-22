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
        """
        Initializes the binning configuration with the given angle and energy ranges, and number of bins.

        Parameters
        ----------
        angleRange : tuple
            (min, max) angle range in degrees
        energyRange : tuple
            (min, max) energy range in MeV
        angleBins : int
            Number of angle bins
        energyBins : int
            Number of energy bins

        Attributes
        ----------
        angleRange : tuple
            (min, max) angle range in degrees
        energyRange : tuple
            (min, max) energy range in MeV
        angleBins : int
            Number of angle bins
        energyBins : int
            Number of energy bins
        angleEdges : numpy.array
            Angle bin edges
        energyEdges : numpy.array
            Energy bin edges
        angleStep : float
            Angle bin step size
        energyStep : float
            Energy bin step size
        """
        self.angleRange = angleRange
        self.energyRange = energyRange
        self.angleBins = angleBins
        self.energyBins = energyBins

        self.angleEdges = np.linspace(angleRange[0], angleRange[1], angleBins + 1)
        self.energyEdges = np.linspace(energyRange[0], energyRange[1], energyBins + 1)

        self.angleStep = self.angleEdges[1] - self.angleEdges[0]
        self.energyStep = self.energyEdges[1] - self.energyEdges[0]

# Global binning config (singleton object)
binningConfig = None

# --------------- TRANSFORMATION VARIABLE ---------------
# This class represents a histogram sampler for sampling angles and energies based on a given histogram.
# It initializes with a histogram, calculates the cumulative distribution function (CDF), and provides a method to sample angles and energies.
class HistogramSampler:
    def __init__(self, hist : np.ndarray, rng=None):
        """
        Initializes the histogram sampler with a given histogram.

        Parameters
        ----------
        hist : numpy.ndarray
            2D histogram with shape (angleBins, energyBins)
        rng : numpy.random.Generator or None
            Random number generator to use for sampling. If None, the default numpy generator is used.

        Attributes
        ----------
        hist : numpy.ndarray
            2D histogram with shape (angleBins, energyBins)
        angleBins : int
            Number of angle bins
        energyBins : int
            Number of energy bins
        rng : numpy.random.Generator
            Random number generator to use for sampling
        flatHist : numpy.ndarray
            1D histogram with shape (angleBins * energyBins,)
        cumsum : numpy.ndarray
            Cumulative sum of the histogram, normalized to 1
        """
        self.hist = hist
        self.angleBins, self.energyBins = hist.shape
        self.rng = rng or np.random.default_rng()

        self.flatHist = hist.flatten()
        self.cumsum = np.cumsum(self.flatHist)
        self.cumsum /= self.cumsum[-1] 

    def sample(self, size=1):
        """
        Samples angles and energies from the histogram.

        Parameters
        ----------
        size : int
            Number of samples to draw

        Returns
        -------
        angles : numpy.ndarray
            Sampled angles with shape (size,)
        energies : numpy.ndarray
            Sampled energies with shape (size,)
        """        
        randValues = self.rng.random(size)
        idxs = np.searchsorted(self.cumsum, randValues, side='right')
        angleIdxs, energyIdxs = np.unravel_index(idxs, (self.angleBins, self.energyBins))

        # Use global binning config
        angles = binningConfig.angleEdges[angleIdxs] + 0.5 * binningConfig.angleStep
        energies = binningConfig.energyEdges[energyIdxs] + 0.5 * binningConfig.energyStep

        return angles, energies

# --------------- TRANSFORMATION VARIABLE ---------------                            
def buildCdfsFromProbTable(probTable : np.ndarray):
    """
    Build cumulative distribution functions (CDFs) from a given probability table.

    Parameters
    ----------
    probTable : numpy.ndarray
        4D array representing the probability table with shape 
        (numMaterials, numEnergies, angleBins, energyBins).

    Returns
    -------
    cdfs : numpy.ndarray
        3D array of CDFs with shape (numMaterials, numEnergies, totalBins)
        where totalBins is the product of angleBins and energyBins.
        Each CDF is normalized to 1.
    """

    numMaterials, numEnergies, angleBins, energyBins = probTable.shape
    totalBins = angleBins * energyBins

    cdfs = np.empty((numMaterials, numEnergies, totalBins), dtype=np.float32)

    for m in range(numMaterials):
        for e in range(numEnergies):
            flatProbTable = probTable[m, e].flatten()
            cdf = np.cumsum(flatProbTable)
            cdf /= cdf[-1] 
            cdfs[m, e] = cdf

    return cdfs

# --------------- TRANSFORMATION VARIABLE ---------------
def prebuildSamplers(data : dict, angleRange : tuple, energyRange : tuple, materialToIndex : dict):
    """
    Prebuilds samplers for all materials and energy bins in a given dataset.

    Parameters
    ----------
    data : dict
        Dictionary containing the dataset
    angleRange : tuple
        Angle range (min, max) in radians
    energyRange : tuple
        Energy range (min, max) in MeV
    materialToIndex : dict
        Dictionary mapping material names to indices

    Notes
    -----
    This function should be called before running the simulation to prebuild the samplers.
    The samplers are stored in a global dictionary, which is cleared each time this function is called.
    The global binning configuration is also set based on the angle and energy ranges and bin counts.
    """
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
                
# --------------- TRANSFORMATION VARIABLE ---------------
@numba.jit(nopython=True, inline = 'always') 
def reverseVariableChangeTransform(initialEnergy, angle, energy):
    """
    Reverse the variable change applied to the energy loss value and angle.

    Parameters:
    - initialEnergy (float): initial energy.
    - angle (float): angle in radians.
    - energy (float): energy loss.

    Returns:
    - realAngle (float): angle after reversing the variable change.
    - realEnergy (float): energy after reversing the variable change.
    """
    realAngle = angle / np.sqrt(initialEnergy)  
    realEnergy = initialEnergy * (1 - np.exp(energy * np.sqrt(initialEnergy)))  
    
    return realAngle, realEnergy                
                
# --------------- TRANSFORMATION VARIABLE ---------------
@numba.jit(nopython=True, inline = 'always') 
def variableChangeTransform(energy, angle, energyloss):
    """
    Applies the variable change to the energy loss value and angle.

    Parameters
    ----------
    energy (float): initial energy.
    angle (float): angle in radians.
    energyloss (float): energy loss.

    Returns
    -------
    energyChange (float): energy change after applying the variable change.
    angleChange (float): angle change after applying the variable change.
    """
    energyChange = np.log((energy - energyloss) / energy) / np.sqrt(energy)
    angleChange = angle * np.sqrt(energy) 
    
    return energyChange, angleChange               
                
# --------------- TRANSFORMATION VARIABLE ---------------
def sampleReverseCalculateInterpolation(data : dict, material : str, energy : float, angleRange : tuple, 
            energyRange : tuple,  materialToIndex : dict):
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
def sampleFromCDFVectorizedTransformation(data : dict, material : str, energies : np.ndarray, materialToIndex : dict, cdfs : np.ndarray):
    """
    Sample from a vectorized array of cumulative distribution functions (CDFs) corresponding to different materials and energy bins.

    Parameters
    ----------
    data : dict
        Dictionary containing the dataset
    material : str
        Material name
    energies : np.ndarray
        Array of energies to sample from
    materialToIndex : dict
        Dictionary mapping material names to indices
    cdfs : np.ndarray
        Array of CDFs corresponding to different materials and energy bins

    Returns
    -------
    sampledAngles : np.ndarray
        Sampled angles
    sampledEnergies : np.ndarray
        Sampled energies
    """
    materialIdx = materialToIndex[material]
    availableEnergies = data['energies']

    # Round energy to match nearest table energy
    roundedEnergies = np.round(energies, 1)
    
    # Check if rounded energies are out of the available range
    minEnergy = np.min(availableEnergies)
    maxEnergy = np.max(availableEnergies)

    # Ensure that rounded energies are within the valid range
    roundedEnergies = np.clip(roundedEnergies, minEnergy, maxEnergy)

    # Find the closest indices for valid energies
    closestIndices = np.array([
        np.argmin(np.abs(availableEnergies - E)) for E in roundedEnergies
    ])

    sampledAngles = np.zeros_like(energies)
    sampledEnergies = np.zeros_like(energies)
    
    angleBins, energyBins = data['probTable'].shape[2:]  # Get from table shape
    binningConfig = BinningConfig(angleRange, energyRange, angleBins, energyBins)

    # Use the CDFs for the closest indices
    uniqueEnergyIndices = np.unique(closestIndices)
    for energyIdx in uniqueEnergyIndices:
        sampleIndices = np.where(closestIndices == energyIdx)[0]
        cdf = cdfs[materialIdx, energyIdx]

        randValues = np.random.random(size=sampleIndices.size)
        flatIdxs = np.searchsorted(cdf, randValues, side='right')
        flatIdxs = np.clip(flatIdxs, 0, len(cdf) - 1)

        angleIdxs, energyIdxs = np.unravel_index(flatIdxs, (angleBins, energyBins))

        realAngles = binningConfig.angleEdges[angleIdxs] + 0.5 * binningConfig.angleStep
        realEnergySamples = binningConfig.energyEdges[energyIdxs] + 0.5 * binningConfig.energyStep
        
        #realAngles = binningConfig.angleEdges[angleIdxs] + np.random.rand(len(angleIdxs)) * binningConfig.angleStep
        #realEnergySamples = binningConfig.energyEdges[energyIdxs] + np.random.rand(len(energyIdxs)) * binningConfig.energyStep

        anglesOut, energiesOut = reverseVariableChangeTransform(
            energies[sampleIndices], realAngles, realEnergySamples
        )

        sampledAngles[sampleIndices] = anglesOut
        sampledEnergies[sampleIndices] = energiesOut

    # Handling energies below the minimum energy
    sampledEnergies[energies < minEnergy] = 0
    sampledAngles[energies < minEnergy] = 0

    # print(f"Energies: {energies}, Rounded Energies: {roundedEnergies}, Closest Indices: {closestIndices}")
    # print(f"Sampled Energies: {sampledEnergies}")
    # print(f"Sampled Angles: {sampledAngles}")

    return sampledAngles, sampledEnergies
                  
# --------------- NORMALIZATION VARIABLE ---------------
def buildCdfsAndBins(data : dict):
    """
    Constructs cumulative distribution functions (CDFs) and bin edges from the given data.

    Parameters
    ----------
    data : dict
        Dictionary containing the following keys:
        - 'probTable': 4D numpy array with shape (numMaterials, numEnergies, angleBins, energyBins) representing the probability table.
        - 'thetaMax': 2D numpy array with shape (numMaterials, numEnergies) for maximum angle values.
        - 'thetaMin': 2D numpy array with shape (numMaterials, numEnergies) for minimum angle values.
        - 'energyMin': 2D numpy array with shape (numMaterials, numEnergies) for minimum energy values.
        - 'energyMax': 2D numpy array with shape (numMaterials, numEnergies) for maximum energy values.

    Returns
    -------
    cdfs : numpy.ndarray
        3D array of CDFs with shape (numMaterials, numEnergies, totalBins), where totalBins is the product of angleBins and energyBins.
    binEdges : dict
        Dictionary mapping (material, energy) tuples to a tuple of numpy arrays (angleEdges, energyEdges) representing the bin edges for angles and energies.
    """
    probTable = data['probTable']
    thetaMax = data['thetaMax']
    thetaMin = data['thetaMin']
    energyMin = data['energyMin']
    energyMax = data['energyMax']
    
    numMaterials, numEnergies, angleBins, energyBins = probTable.shape
    totalBins = angleBins * energyBins
    cdfs = np.zeros((numMaterials, numEnergies, totalBins), dtype=np.float32)
    binEdges = {}
    
    for m in range(numMaterials):
        for e in range(numEnergies):
            histFlat = probTable[m, e].flatten()
            cdf = np.cumsum(histFlat)
            if cdf[-1] == 0:
                # If the CDF sum is zero, set all values in the CDF to zero
                cdfs[m, e] = np.zeros_like(cdf) 
                binEdges[(m, e)] = (None, None)
            else:
                cdf /= cdf[-1]
                cdfs[m, e] = cdf
            
            # Create bin edges for material and energy
            thetaMaxValue = thetaMax[m, e]
            thetaMinValue = thetaMin[m, e]
            energyMinValue = energyMin[m, e]
            energyMaxValue = energyMax[m, e]
            
            angleEdges = np.linspace(thetaMinValue, thetaMaxValue, angleBins + 1)
            energyEdges = np.linspace(energyMinValue, energyMaxValue, energyBins + 1)

            binEdges[(m, e)] = (angleEdges, energyEdges)
            
    return cdfs, binEdges

# --------------- NORMALIZATION VARIABLE ---------------
def serializeBinEdges(binEdges : dict, numMaterials : int, numEnergies : int, angleBins : int, energyBins : int): 
    """
    Serialize bin edges for all material and energy combinations into two arrays that can be shared with workers.

    Parameters:
    - binEdges (dict): Dictionary mapping (material, energy) tuples to a tuple of numpy arrays (angleEdges, energyEdges) representing the bin edges for angles and energies.
    - numMaterials (int): Number of materials.
    - numEnergies (int): Number of energy bins.
    - angleBins (int): Number of angle bins.
    - energyBins (int): Number of energy bins.

    Returns:
    - angleEdgesArray (ndarray): 3D array of shape (numMaterials, numEnergies, angleBins + 1) containing the angle bin edges.
    - energyEdgesArray (ndarray): 3D array of shape (numMaterials, numEnergies, energyBins + 1) containing the energy bin edges.
    """
    angleEdgesArray = np.zeros((numMaterials, numEnergies, angleBins + 1), dtype=np.float32)
    energyEdgesArray = np.zeros((numMaterials, numEnergies, energyBins + 1), dtype=np.float32)
    
    for m in range(numMaterials):
        for e in range(numEnergies):
            angleEdges, energyEdges = binEdges[(m, e)]
            angleEdgesArray[m, e] = angleEdges
            energyEdgesArray[m, e] = energyEdges
            
    return angleEdgesArray, energyEdgesArray

# --------------- NORMALIZATION VARIABLE ---------------
def reconstructBinEdges(angleEdgesArray : np.ndarray, energyEdgesArray : np.ndarray):
    """
    Reconstructs the binEdges dictionary from the serialized binEdges arrays.

    Parameters:
    - angleEdgesArray (ndarray): 3D array of shape (numMaterials, numEnergies, angleBins + 1) containing the angle bin edges.
    - energyEdgesArray (ndarray): 3D array of shape (numMaterials, numEnergies, energyBins + 1) containing the energy bin edges.

    Returns:
    - binEdges (dict): Dictionary mapping (material, energy) tuples to a tuple of numpy arrays (angleEdges, energyEdges) representing the bin edges for angles and energies.
    """
    numMaterials, numEnergies, _ = angleEdgesArray.shape
    binEdges = {}
    for m in range(numMaterials):
        for e in range(numEnergies):
            binEdges[(m, e)] = (angleEdgesArray[m, e], energyEdgesArray[m, e])
    return binEdges

# --------------- NORMALIZATION VARIABLE ---------------
@numba.jit(nopython=True, inline = 'always') 
def reverseVariableChangeNormalized(normalizedAngle, normalizedEnergy, thetaMax, thetaMin, energyMin, energyMax):
    """
    Reverse the normalization to physical values.

    Parameters:
    - normalizedAngle (float): Normalized angle between 0 and 1.
    - normalizedEnergy (float): Normalized energy between 0 and 1.
    - thetaMax (float): Maximum angle in radians.
    - thetaMin (float): Minimum angle in radians.
    - energyMax (float): Maximum energy in MeV.
    - energyMin (float): Minimum energy in MeV.

    Returns:
    - realAngle (float): Angle in radians after reversing the normalization.
    - realEnergy (float): Energy in MeV after reversing the normalization.
    """
    realAngle = normalizedAngle * (thetaMax - thetaMin) + thetaMin
    realEnergy = energyMin + normalizedEnergy * (energyMax - energyMin)
    return realAngle, realEnergy

# --------------- NORMALIZATION VARIABLE ---------------
def sampleFromCDFVectorizedNormalization(data : dict, material : str, energies : np.ndarray, materialToIndex : dict,
        cdfs : np.ndarray,  binEdges : dict):
    """
    Samples from a vectorized array of cumulative distribution functions (CDFs) corresponding to different materials and energy bins.

    Parameters
    ----------
    data : dict
        Dictionary containing the dataset
    material : str
        Material name
    energies : np.ndarray
        Array of energies to sample from
    materialToIndex : dict
        Dictionary mapping material names to indices
    cdfs : np.ndarray
        Array of CDFs corresponding to different materials and energy bins
    binEdges : dict
        Dictionary mapping (material, energy) tuples to a tuple of numpy arrays (angleEdges, energyEdges) representing the bin edges for angles and energies

    Returns
    -------
    sampledAngles : np.ndarray
        Sampled angles
    sampledEnergies : np.ndarray
        Sampled energies
    """
    materialIdx = materialToIndex[material]
    availableEnergies = data['energies']
    thetaMax = data['thetaMax']
    thetaMin = data['thetaMin']
    energyMin = data['energyMin']
    energyMax = data['energyMax']
    angleBins, energyBins = data['probTable'].shape[2:]

    sampledAngles = np.zeros_like(energies)
    sampledEnergies = np.zeros_like(energies)

    minEnergy = np.min(availableEnergies)
    roundedEnergies = np.round(energies, 1)

    closestIndices = np.array([
        np.argmin(np.abs(availableEnergies - E)) for E in roundedEnergies
    ])

    uniqueEnergyIndices = np.unique(closestIndices)

    for energyIdxInAvailable in uniqueEnergyIndices:
        idxs = np.where(closestIndices == energyIdxInAvailable)[0]
        
        # Check if binEdges[(materialIdx, energyIdxInAvailable)] is (None, None)
        angleEdges, energyEdges = binEdges.get((materialIdx, energyIdxInAvailable), (None, None))

        # If bin edges are None, return 0 for all sampled angles and energies
        if angleEdges is None or energyEdges is None:
            sampledAngles[idxs] = 0.0
            sampledEnergies[idxs] = 0.0
            continue

        cdf = cdfs[materialIdx, energyIdxInAvailable]

        rand = np.random.random(size=idxs.size)
        flatIdxs = np.searchsorted(cdf, rand, side='right')
        flatIdxs = np.clip(flatIdxs, 0, len(cdf) - 1)

        angleIdxs, energyIdxs = np.unravel_index(flatIdxs, (angleBins, energyBins))

        # normalizedAngles = (angleIdxs + 0.5) / angleBins  # center of bin as fraction
        # normalizedEnergies = (energyIdxs + 0.5) / energyBins

        # thetaMaxVal = thetaMax[materialIdx, energyIdxInAvailable]
        # thetaMinVal = thetaMin[materialIdx, energyIdxInAvailable]
        # energyMinVal = energyMin[materialIdx, energyIdxInAvailable]
        # energyMaxVal = energyMax[materialIdx, energyIdxInAvailable]

        # anglesOut, energiesOut = reverseVariableChangeNormalized(
        #     normalizedAngles, normalizedEnergies,
        #     thetaMaxVal, thetaMinVal,
        #     energyMinVal, energyMaxVal
        # )
        # sampledAngles[idxs] = anglesOut
        # sampledEnergies[idxs] = energiesOut
        
        angleBinWidths = np.diff(angleEdges)
        energyBinWidths = np.diff(energyEdges)

        sampledAngles[idxs] = angleEdges[angleIdxs] + np.random.rand(len(idxs)) * angleBinWidths[angleIdxs]
        sampledEnergies[idxs] = energyEdges[energyIdxs] + np.random.rand(len(idxs)) * energyBinWidths[energyIdxs]
        
        #angleCenters = (angleEdges[:-1] + angleEdges[1:]) / 2
        #energyCenters = (energyEdges[:-1] + energyEdges[1:]) / 2

        #sampledAngles[idxs] = angleCenters[angleIdxs]
        #sampledEnergies[idxs] = energyCenters[energyIdxs]
        
        #sampledAngles[idxs] = angleEdges[angleIdxs]
        #sampledEnergies[idxs] = energyEdges[energyIdxs]
        
    # Identify invalid (too low) energies
    mask = roundedEnergies <= minEnergy

    # Assign 0 to all invalid energies directly
    sampledAngles[mask] = 0.0
    sampledEnergies[mask] = 0.0

    return sampledAngles, sampledEnergies

# --------------- COMMON FUNCTIONS ----------------
def createPhysicalSpace(bigVoxel : tuple,  voxelShapeBins : tuple, dt = 1 / 3):
    """
    Create physical space coordinates for 3D binning.

    Parameters
    ----------
    bigVoxel : tuple
        Size of the physical space in each dimension.
    voxelShapeBins : tuple
        Number of bins in each dimension.
    dt : float, optional
        Time step size. Defaults to 1/3.

    Returns
    -------
    xRange, yRange, zRange : tuple
        3 tuples of coordinates for each dimension.
    """
    xRange = np.linspace(-bigVoxel[0] / dt, bigVoxel[0] / dt, voxelShapeBins[0]) 
    yRange = np.linspace(-bigVoxel[1] / dt, bigVoxel[1] / dt, voxelShapeBins[1])
    zRange = np.linspace(-bigVoxel[2] / dt, bigVoxel[2] / dt, voxelShapeBins[2])
    
    return xRange, yRange, zRange

# --------------- COMMON FUNCTIONS ----------------
@numba.jit(nopython=True)
def calculateEnergyDepositBinBatch(positions : np.ndarray, physicalSize : tuple, energyLosses : np.ndarray, 
            energyDepositedVector : np.ndarray, voxelShapeBins : tuple):
    """
    Numba-compatible fast energy deposition.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of positions of particles
    physicalSize : tuple
        Size of the physical space in each dimension
    energyLosses : np.ndarray
        Array of energy losses of particles
    energyDepositedVector : np.ndarray
        The array to store the deposited energy in each voxel
    voxelShapeBins : tuple
        Number of bins in each dimension
    
    Returns
    -------
    energyDepositedVector : np.ndarray
        The array with the deposited energy in each voxel
    """
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
    
        #print(f"Energy deposited in voxel bin {ix}, {iy}, {iz}:", energyDepositedVector[ix, iy, iz])
        
    return energyDepositedVector

# --------------- COMMON FUNCTIONS ----------------
def sampleFromCDF(
    data, material, energies, materialToIndex, cdfs, binEdges=None, method="transformation"
):
    """
    Sample angles and energies from a given material and energy using a given method.

    Parameters
    ----------
    data : dict
        Data dictionary containing the energy-dependent scattering properties.
    material : str
        Material to sample from.
    energies : np.ndarray
        Array of energies to sample from.
    materialToIndex : dict
        Mapping of material names to indices in the data dictionary.
    cdfs : np.ndarray
        Cumulative distribution functions for the given material and energy.
    binEdges : tuple of np.ndarray, optional
        Bin edges for the given material and energy. Must be provided for 'normalization' method.
    method : str, optional
        Sampling method to use. Can be 'transformation' or 'normalization'.

    Returns
    -------
    sampledAngles, sampledEnergies : tuple of np.ndarray
        Arrays of sampled angles and energies.
    """
    if method == "transformation":
        return sampleFromCDFVectorizedTransformation(
            data=data,
            material=material,
            energies=energies,
            materialToIndex=materialToIndex,
            cdfs=cdfs
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
        
# --------------- COMMON FUNCTIONS ----------------
def simulateBatchParticlesVectorized(
    batchSize, data, material, initialEnergy,
    materialToIndex, bigVoxelSize, 
    energyDepositedVector, cdfs, 
    binEdges = None,
    method = 'transformation'
):
    """
    Simulate a batch of particles using vectorized operations for efficiency.

    This function initializes particle positions, velocities, and energies, and iteratively updates them based on scattering and energy loss. 
    The simulation continues until all particles are either out of bounds or have zero energy. The energy deposited in the voxel is calculated and stored. 
    The function uses either a 'transformation' or 'normalization' method to sample angles and energies.

    Parameters
    ----------
    batchSize : int
        Number of particles in the batch.
    data : dict
        Dictionary containing the dataset with energy-dependent scattering properties.
    material : str
        Material to simulate particles through.
    initialEnergy : float
        Initial energy of the particles.
    materialToIndex : dict
        Mapping from material names to indices for data access.
    bigVoxelSize : tuple
        Size of the voxel in the simulation space.
    energyDepositedVector : np.ndarray
        Array to store the deposited energy in each voxel bin.
    cdfs : np.ndarray
        Array of cumulative distribution functions for sampling.
    binEdges : dict, optional
        Dictionary of bin edges for 'normalization' method.
    method : str, optional
        Sampling method, either 'transformation' or 'normalization'. Defaults to 'transformation'.
    """
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
                materialToIndex=materialToIndex, cdfs=cdfs, binEdges=binEdges, method=method)

        energyLossPerStep = energyActive - realEnergies
        energy[active] = realEnergies
        
        # dtScaled = np.ones_like(energyActive) * dt
        # lowEnergyMask = energyActive < energyThreshold
        # k = kMin + (kMax - kMin) * (energyActive[lowEnergyMask] / energyThreshold)
        # dtScaled[lowEnergyMask] *= k
    
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
        nearZero = norms[:, 0] < 1e-8
        perp[nearZero] = np.array([1.0, 0.0, 0.0])
        perp[~nearZero] /= norms[~nearZero]

        crossPerpV = np.cross(perp, v)
        cosTheta = np.cos(theta)[:, np.newaxis]
        sinTheta = np.sin(theta)[:, np.newaxis]
        cosPhi = np.cos(phi)[:, np.newaxis]
        sinPhi = np.sin(phi)[:, np.newaxis]

        w = cosTheta * v + sinTheta * (cosPhi * crossPerpV + sinPhi * perp)
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        velocity[active] = w
        position[active] += velocity[active] * dt
        # position[active] += velocity[active] * dtScaled[:, np.newaxis]

        # Check if particles are still active
        withinBounds = np.all(
            (position[active] >= -bigVoxelSize) & (position[active] <= bigVoxelSize),
            axis=1
        )
        realEnergiesValid = realEnergies > 0
        activeIndices = np.where(active)[0]
        active[activeIndices] = realEnergiesValid & withinBounds
        
# --------------- COMMON FUNCTIONS ----------------
def simulateBatchParticlesWorker(args):
    """
    Simulate particles in parallel using shared memory and multiprocessing.

    This function is designed to work as a multiprocessing worker, handling
    the initialization and simulation of a batch of particles. It reconstructs
    necessary data from shared memory and calls `simulateBatchParticlesVectorized`
    to perform the actual simulation.

    Parameters
    ----------
    args : tuple
        A tuple containing all necessary arguments. The arguments vary based
        on the selected sampling method ('normalization' or 'transformation').
        Common elements include:
        - Shared memory names and shapes for probability tables, CDFs, and
          energy deposition vectors.
        - Batch size, materials, energies, material, initial energy,
          material-to-index mapping, big voxel size, seed, and method.

    Raises
    ------
    ValueError
        If an unknown sampling method is specified.
    None
    """
    method = args[17] 

    if method == 'normalization':
        (
            shm_name, shape, dtype_str,
            shm_cdfs_name, cdfs_shape, cdfs_dtype_str,
            shm_energy_deposited_name, energy_deposited_shape, energy_deposited_dtype_str,
            batchSize, materials, energies,
            material, initialEnergy,
            materialToIndex, bigVoxelSize,
            seed, method,
            shm_angle_edges_name, angle_edges_shape, angle_edges_dtype_str,
            shm_energy_edges_name, energy_edges_shape, energy_edges_dtype_str,
            shm_theta_max_name, thetaMax_shape, thetaMax_dtype_str,
            shm_theta_min_name, thetaMin_shape, thetaMin_dtype_str,
            shm_energy_min_name, energyMin_shape, energyMin_dtype_str,
            shm_energy_max_name, energyMax_shape, energyMax_dtype_str,
        ) = args
    elif method == 'transformation':
        (
            shm_name, shape, dtype_str,
            shm_cdfs_name, cdfs_shape, cdfs_dtype_str,
            shm_energy_deposited_name, energy_deposited_shape, energy_deposited_dtype_str,
            batchSize, materials, energies,
            material, initialEnergy,
            materialToIndex, bigVoxelSize,
            seed, method
        ) = args
        # Set unused shared memory references to None for consistency
        shm_angle_edges_name = angle_edges_shape = angle_edges_dtype_str = None
        shm_energy_edges_name = energy_edges_shape = energy_edges_dtype_str = None
        shm_theta_max_name = thetaMax_shape = thetaMax_dtype_str = None
        shm_theta_min_name = thetaMin_shape = thetaMin_dtype_str = None
        shm_energy_min_name = energyMin_shape = energyMin_dtype_str = None
        shm_energy_max_name = energyMax_shape = energyMax_dtype_str = None
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    # Seed for reproducibility
    np.random.seed(seed)
    
    # Attach to shared memory for prob_table
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    prob_table = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=existing_shm.buf)
    
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
    
    binEdges = None
    if method == 'normalization':
        # Attach to shared memory for angle edges and energy edges
        existing_shm_angle_edges = shared_memory.SharedMemory(name=shm_angle_edges_name)
        angleEdges = np.ndarray(angle_edges_shape, dtype=np.dtype(angle_edges_dtype_str), buffer=existing_shm_angle_edges.buf)
        existing_shm_energy_edges = shared_memory.SharedMemory(name=shm_energy_edges_name)
        energyEdges = np.ndarray(energy_edges_shape, dtype=np.dtype(energy_edges_dtype_str), buffer=existing_shm_energy_edges.buf)
        
        # Attach to shared memory for thetaMax and thetaMin and energyMin and energyMax values
        existing_shm_theta_max = shared_memory.SharedMemory(name=shm_theta_max_name)
        thetaMax = np.ndarray(thetaMax_shape, dtype=np.dtype(thetaMax_dtype_str), buffer=existing_shm_theta_max.buf)
        existing_shm_theta_min = shared_memory.SharedMemory(name=shm_theta_min_name)
        thetaMin = np.ndarray(thetaMin_shape, dtype=np.dtype(thetaMin_dtype_str), buffer=existing_shm_theta_min.buf)
        existing_shm_energy_min = shared_memory.SharedMemory(name=shm_energy_min_name)
        energyMin = np.ndarray(energyMin_shape, dtype=np.dtype(energyMin_dtype_str), buffer=existing_shm_energy_min.buf)
        existing_shm_energy_max = shared_memory.SharedMemory(name=shm_energy_max_name)
        energyMax = np.ndarray(energyMax_shape, dtype=np.dtype(energyMax_dtype_str), buffer=existing_shm_energy_max.buf)
        
        data.update({
                'thetaMax': thetaMax,
                'thetaMin': thetaMin,
                'energyMin': energyMin,
                'energyMax': energyMax
            })
        binEdges = reconstructBinEdges(angleEdges, energyEdges)
        
    elif method == 'transformation':
        pass # No extra shared memory needed
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    return simulateBatchParticlesVectorized(batchSize=batchSize, data=data, material=material, initialEnergy=initialEnergy,
            materialToIndex=materialToIndex, bigVoxelSize=bigVoxelSize, energyDepositedVector=energyDepositedVector, cdfs=cdfs,
            binEdges=binEdges, method=method)
    

# --------------- COMMON FUNCTIONS ----------------
def simulateBatch(args):
    """
    Simulate particles in parallel using shared memory and multiprocessing.

    This function is a simple wrapper around `simulateBatchParticlesWorker` that
    calls it with the provided arguments.

    Parameters
    ----------
    args : tuple
        A tuple containing all necessary arguments. The arguments vary based
        on the selected sampling method ('normalization' or 'transformation').
        Common elements include:
        - Shared memory names and shapes for probability tables, CDFs, and
          energy deposition vectors.
        - Batch size, materials, energies, material, initial energy,
          material-to-index mapping, big voxel size, seed, and method.
    """
    return simulateBatchParticlesWorker(args)

# --------------- COMMON FUNCTIONS ----------------
def runMultiprocessedBatchedSim(
    totalSamples, batchSize, numWorkers,
    shm_prob_table, prob_table_shape, prob_table_dtype,
    shm_cdfs, cdfs_shape, cdfs_dtype,
    shm_energy_deposited, energy_deposited_shape, energy_deposited_dtype,
    data, material, initialEnergy, materialToIndex, 
    bigVoxelSize, method = 'transformation',
    # Optional shared memory for normalization
    shm_angle_edges = None, angle_edges_shape = None, angle_edges_dtype = None,
    shm_energy_edges = None, energy_edges_shape = None, energy_edges_dtype = None,
    shm_theta_max = None, thetaMax_shape = None, thetaMax_dtype = None,
    shm_theta_min = None, thetaMin_shape = None, thetaMin_dtype = None,
    shm_energy_min = None, energyMin_shape = None, energyMin_dtype = None,
    shm_energy_max = None, energyMax_shape = None, energyMax_dtype = None,
):
    """
    Run a simulation in parallel across multiple workers using shared memory.

    This function is responsible for:

    - Dividing the total number of samples into batches
    - Creating a list of arguments for each worker with chunked tasks
    - Optionally, creating a list of arguments for normalization
    - Running the simulation in parallel using a process pool
    - Collecting the results

    Parameters
    ----------
    totalSamples : int
        Total number of samples to simulate
    batchSize : int
        Number of samples to simulate in a single batch
    numWorkers : int
        Number of worker processes to use
    shm_prob_table : multiprocessing.shared_memory.SharedMemory
        Shared memory object for probability tables
    shm_prob_table_shape : tuple
        Shape of the probability tables
    shm_prob_table_dtype : numpy.dtype
        Data type of the probability tables
    shm_cdfs : multiprocessing.shared_memory.SharedMemory
        Shared memory object for CDFs
    shm_cdfs_shape : tuple
        Shape of the CDFs
    shm_cdfs_dtype : numpy.dtype
        Data type of the CDFs
    shm_energy_deposited : multiprocessing.shared_memory.SharedMemory
        Shared memory object for energy deposition vectors
    shm_energy_deposited_shape : tuple
        Shape of the energy deposition vectors
    shm_energy_deposited_dtype : numpy.dtype
        Data type of the energy deposition vectors
    data : dict
        Dictionary containing necessary data
    material : str
        Material to simulate
    initialEnergy : float
        Initial energy of the simulation
    materialToIndex : dict
        Dictionary mapping material names to indices
    bigVoxelSize : float
        Size of the big voxel
    method : str (optional)
        Method to use for sampling. If not provided, defaults to 'transformation'
    shm_angle_edges : multiprocessing.shared_memory.SharedMemory (optional)
        Shared memory object for angle edges
    angle_edges_shape : tuple (optional)
        Shape of the angle edges
    angle_edges_dtype : numpy.dtype (optional)
        Data type of the angle edges
    shm_energy_edges : multiprocessing.shared_memory.SharedMemory (optional)
        Shared memory object for energy edges
    energy_edges_shape : tuple (optional)
        Shape of the energy edges
    energy_edges_dtype : numpy.dtype (optional)
        Data type of the energy edges
    shm_theta_max : multiprocessing.shared_memory.SharedMemory (optional)
        Shared memory object for theta max
    thetaMax_shape : tuple (optional)
        Shape of theta max
    thetaMax_dtype : numpy.dtype (optional)
        Data type of theta max
    shm_theta_min : multiprocessing.shared_memory.SharedMemory (optional)
        Shared memory object for theta min
    thetaMin_shape : tuple (optional)
        Shape of theta min
    thetaMin_dtype : numpy.dtype (optional)
        Data type of theta min
    shm_energy_min : multiprocessing.shared_memory.SharedMemory (optional)
        Shared memory object for energy min
    energyMin_shape : tuple (optional)
        Shape of energy min
    energyMin_dtype : numpy.dtype (optional)
        Data type of energy min
    shm_energy_max : multiprocessing.shared_memory.SharedMemory (optional)
        Shared memory object for energy max
    energyMax_shape : tuple (optional)
        Shape of energy max
    energyMax_dtype : numpy.dtype (optional)
        Data type of energy max
    """
    # Number of batches to process
    numBatches = (totalSamples + batchSize - 1) // batchSize
    argsList = []
    
    # Create a list of arguments for each worker with chunked tasks
    for i in range(numBatches):
        args = [
            shm_prob_table.name, prob_table_shape, prob_table_dtype.name,
            shm_cdfs.name, cdfs_shape, cdfs_dtype.name,
            shm_energy_deposited.name, energy_deposited_shape, energy_deposited_dtype.name,
            min(batchSize, totalSamples - i * batchSize),
            data['materials'], data['energies'], 
            material, initialEnergy, materialToIndex,
            bigVoxelSize, 1337 + i,
            method,
        ]

        if method == 'normalization':
            args.extend([
                shm_angle_edges.name, angle_edges_shape, angle_edges_dtype.name,
                shm_energy_edges.name, energy_edges_shape, energy_edges_dtype.name,
                shm_theta_max.name, thetaMax_shape, thetaMax_dtype.name,
                shm_theta_min.name, thetaMin_shape, thetaMin_dtype.name,
                shm_energy_min.name, energyMin_shape, energyMin_dtype.name,
                shm_energy_max.name, energyMax_shape, energyMax_dtype.name,
            ])

        argsList.append(tuple(args))
            
    chunkSize = 1
    with Pool(processes=numWorkers) as pool:
        list(tqdm(pool.imap_unordered(simulateBatchParticlesWorker, argsList, chunksize=chunkSize), total=len(argsList)))
        
# --------------- COMMON FUNCTIONS ----------------
def plotSamplingDistribution(data : dict, material : str, fixedEnergyIdx : int, materialToIndex : dict, cdfs : np.ndarray, 
            method='transformation', N=10000, binEdges=None):
    """
    Sample angles and energies from the given material and fixed energy index,
    using either the transformation or normalization methods. The sampled values
    are saved to a CSV file and a histogram is plotted and saved as a PDF file.

    Parameters
    ----------
    data : dict
        Dictionary containing the dataset
    material : str
        Material name
    fixedEnergyIdx : int
        Index of the fixed energy to sample
    materialToIndex : dict
        Dictionary mapping material names to indices
    cdfs : np.ndarray
        Array of CDFs corresponding to different materials and energy bins
    method : str, optional
        Sampling method to use, either 'transformation' or 'normalization'
    N : int, optional
        Number of samples to generate
    binEdges : tuple, optional
        Tuple of numpy arrays (angleEdges, energyEdges) representing the bin edges for angles and energies,
        used only if method is 'normalization'
    """
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
    numberOfBins = 100
    material = 'G4_WATER'
    initialEnergy = 200.0  # MeV
    bigVoxelSize = np.array((33.3333, 33.33333, 50), dtype=np.float64)
    voxelShapeBins = (50, 50, 300)
    dt = 1 / 3
    
    angleRange = (0, 70)
    energyRange = (-0.6, 0)

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
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(npyPath, exist_ok=True)
    os.makedirs(csvPath, exist_ok=True)

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
        cdfs, binEdges = buildCdfsAndBins(data)
        angleEdgeArray, energyEdgeArray = serializeBinEdges(binEdges, len(materials), len(energies), numberOfBins, numberOfBins)
        # energy = 15.
        # idx = int(np.argmin(np.abs(energies - energy)))
        # plotSamplingDistribution(data, material, idx, materialToIndex, cdfs, method=method, N=10000000, binEdges=binEdges)
    else:
        cdfs = buildCdfsFromProbTable(probTable)
        binEdges = None
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
    
    # Shared memory for cfds
    shm_cdfs = shared_memory.SharedMemory(create=True, size=cdfs.nbytes)
    shm_cdfs_np = np.ndarray(cdfs.shape, dtype=cdfs.dtype, buffer=shm_cdfs.buf)
    np.copyto(shm_cdfs_np, cdfs)
    
    # Create shared memory for the table
    shm_prob_table = shared_memory.SharedMemory(create=True, size=probTable.nbytes)
    shm_prob_np = np.ndarray(probTable.shape, dtype=probTable.dtype, buffer=shm_prob_table.buf)
    np.copyto(shm_prob_np, probTable)
    
    # Create shared memory for the energy deposited
    energyDeposited = np.zeros(voxelShapeBins, dtype=np.float32)
    shm_energy_deposited = shared_memory.SharedMemory(create=True, size=energyDeposited.nbytes)
    shm_energy_deposited_np = np.ndarray(energyDeposited.shape, dtype=energyDeposited.dtype, buffer=shm_energy_deposited.buf)
    np.copyto(shm_energy_deposited_np, energyDeposited)
    
    # Get shapes/dtypes for shared arrays
    prob_table_shape = probTable.shape
    prob_table_dtype = probTable.dtype
    
    batchSize = 1000
    numWorkers = cpu_count()
    
    kwargs = {}
    
    if method == 'normalization':
        thetaMax = data['thetaMax']
        thetaMin = data['thetaMin']
        energyMin = data['energyMin']
        energyMax = data['energyMax']
        
        # Create shared memory and fill it for normalization-specific arrays
        shm_angle_edges = shared_memory.SharedMemory(create=True, size=angleEdgeArray.nbytes)
        shm_energy_edges = shared_memory.SharedMemory(create=True, size=energyEdgeArray.nbytes)
        shm_angle_np = np.ndarray(angleEdgeArray.shape, dtype=angleEdgeArray.dtype, buffer=shm_angle_edges.buf)
        shm_energy_np = np.ndarray(energyEdgeArray.shape, dtype=energyEdgeArray.dtype, buffer=shm_energy_edges.buf)
        np.copyto(shm_angle_np, angleEdgeArray)
        np.copyto(shm_energy_np, energyEdgeArray)

        shm_theta_max = shared_memory.SharedMemory(create=True, size=thetaMax.nbytes)
        theta_max_shm_array = np.ndarray(thetaMax.shape, dtype=thetaMax.dtype, buffer=shm_theta_max.buf)
        theta_max_shm_array[:] = thetaMax[:]

        shm_theta_min = shared_memory.SharedMemory(create=True, size=thetaMin.nbytes)
        theta_min_shm_array = np.ndarray(thetaMin.shape, dtype=thetaMin.dtype, buffer=shm_theta_min.buf)
        theta_min_shm_array[:] = thetaMin[:]

        shm_energy_min = shared_memory.SharedMemory(create=True, size=energyMin.nbytes)
        energy_min_shm_array = np.ndarray(energyMin.shape, dtype=energyMin.dtype, buffer=shm_energy_min.buf)
        energy_min_shm_array[:] = energyMin[:]

        shm_energy_max = shared_memory.SharedMemory(create=True, size=energyMax.nbytes)
        energy_max_shm_array = np.ndarray(energyMax.shape, dtype=energyMax.dtype, buffer=shm_energy_max.buf)
        energy_max_shm_array[:] = energyMax[:]
    
        # Add normalization-specific arguments to kwargs
        kwargs.update(dict(
            shm_angle_edges=shm_angle_edges,
            angle_edges_shape=angleEdgeArray.shape,
            angle_edges_dtype=angleEdgeArray.dtype,
            shm_energy_edges=shm_energy_edges,
            energy_edges_shape=energyEdgeArray.shape,
            energy_edges_dtype=energyEdgeArray.dtype,
            shm_theta_max=shm_theta_max,
            thetaMax_shape=thetaMax.shape,
            thetaMax_dtype=thetaMax.dtype,
            shm_theta_min=shm_theta_min,
            thetaMin_shape=thetaMin.shape,
            thetaMin_dtype=thetaMin.dtype,
            shm_energy_min=shm_energy_min,
            energyMin_shape=energyMin.shape,
            energyMin_dtype=energyMin.dtype,
            shm_energy_max=shm_energy_max,
            energyMax_shape=energyMax.shape,
            energyMax_dtype=energyMax.dtype,
        ))
    
    # Run simulation
    print(f"Running simulation using '{method}' method.")
    shm_energy_deposited_np.fill(0)

    runMultiprocessedBatchedSim(
        samplingN, batchSize, numWorkers,
        shm_prob_table, prob_table_shape, prob_table_dtype,
        shm_cdfs, cdfs.shape, cdfs.dtype,
        shm_energy_deposited, energyDeposited.shape, energyDeposited.dtype,
        data, material, initialEnergy, materialToIndex,
        bigVoxelSize, method,
        **kwargs  # Pass extra args depending on method
    )
                    
    energyVector3D = np.ndarray(energyDeposited.shape, dtype=energyDeposited.dtype, buffer=shm_energy_deposited.buf).copy()
    np.save(f'{npyPath}projectionXZSimulation_{method}.npy', energyVector3D)
        
    x, y, z = np.nonzero(energyVector3D)
    energies = energyVector3D[x, y, z]

    energyGrid = np.zeros(voxelShapeBins)

    for xi, yi, zi, ei in zip(x, y, z, energies):
        energyGrid[xi, yi, zi] = ei

    projectionXZ = np.sum(energyGrid, axis=1)  # axis=1 is Y

    # Create coordinate ranges
    xRange, _, zRange = createPhysicalSpace(bigVoxelSize, voxelShapeBins)

    # Plot the projection
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
            projectionXZ.T,
            extent=[xRange[0], xRange[-1], zRange[0], zRange[-1]],
            origin='lower',
            aspect='auto',
            cmap='Blues'
    )
    ax.axhline(y=110, color='red', linestyle='--', linewidth=1.5, label='Z = 100 mm')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Summed Energy Deposit (MeV)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')

    plt.tight_layout()
    plt.savefig(f"{savePath}EnergyDeposit_XZ_ProjectionSimulation_{method}.pdf", dpi=300)
    plt.close(fig)
                
    # Profile of the beam at X = 0 and X-Axis
    indxCenter = energyVector3D.shape[0] // 2
    profileZ = energyVector3D[indxCenter, :, :]
    profileZMean = profileZ.mean(axis=0)
                
    zIndex = 50
    profileX = energyVector3D[:, :, zIndex]
    profileXMean = profileX.mean(axis=1)

    fig1, ax1 =plt.subplots(1, 2, figsize=(10, 6))
    ax1[0].plot(zRange, profileZMean)
    ax1[0].set_xlabel(r'Z voxel Index')
    ax1[0].set_ylabel(r'Energy Deposit (MeV)')
    # ax1[0].set_xlim(- bigVoxelSize[2] / dt, + bigVoxelSize[2] / dt)
                
    ax1[1].plot(xRange, profileXMean)
    ax1[1].set_xlabel(r'X voxel Index')
    ax1[1].set_ylabel(r'Energy Deposit (MeV)')
    # ax1[1].set_xlim(-bigVoxelSize[0] / dt, + bigVoxelSize[0] / dt)
                
    plt.tight_layout()
    plt.savefig(f'{savePath}ProfilesEnergyDepositSimulation_{method}.pdf')
    plt.close(fig1)
        
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
        shm_angle_edges.close()
        shm_angle_edges.unlink()
        shm_energy_edges.close()
        shm_energy_edges.unlink()
        
        shm_theta_max.close()
        shm_theta_max.unlink()
        shm_theta_min.close()
        shm_theta_min.unlink()
        shm_energy_min.close()
        shm_energy_min.unlink()
        shm_energy_max.close()
        shm_energy_max.unlink()
        
    endTime = time.time()
    print(f"Simulation time: {endTime - startTime:.12f} seconds")
    print()