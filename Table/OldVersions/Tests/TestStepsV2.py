import numpy as np
import numba
from numba import prange
from collections import defaultdict

@numba.jit(nopython=True)
def getVoxelIndex(position, size, bins):
    # This is already good and JIT-compiled
    ix = int((position[0] + size[0]) * bins[0] / (2 * size[0]))
    iy = int((position[1] + size[1]) * bins[1] / (2 * size[1]))
    iz = int((position[2] + size[2]) * bins[2] / (2 * size[2]))

    ix = min(max(ix, 0), bins[0] - 1)
    iy = min(max(iy, 0), bins[1] - 1)
    iz = min(max(iz, 0), bins[2] - 1)

    return ix, iy, iz

# Keeping this as a regular Python function or JIT-compile if needed elsewhere
def getMaterialIndexAtPosition(positions, grid, physicalSize, voxelShapeBins):
    n = positions.shape[0]
    indices = np.empty(n, dtype=np.int32)

    for i in range(n):
        ix, iy, iz = getVoxelIndex(positions[i], physicalSize, voxelShapeBins)
        indices[i] = grid[ix, iy, iz]

    return indices

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

            # Store voxel and segment info if inside grid:
            ix, iy, iz = voxel
            if 0 <= ix < binsX and 0 <= iy < binsY and 0 <= iz < binsZ:
                material = grid[ix, iy, iz]
                materialIndices[i, count] = material
                segmentLengths[i, count] = travelLength
                voxelIndices[i, count, :] = voxel
                # print(f'Travel new segment:', count, 'at voxel:', ix, iy, iz, 'length:', travelLength, 'material:', material)
                count += 1
            else:
                # Out of bounds voxel - stop traversal for this particle
                break

            traveled += travelLength

            # Move to next voxel along axis:
            voxel[axis] += step[axis]
            tMax[axis] += tDelta[axis]

        segmentCounts[i] = count

    return materialIndices, segmentLengths, segmentCounts

@numba.jit(parallel=True, fastmath=True)
def depositEnergy3DStepTraversal(
    voxelIndices, segmentLengths, segmentCounts, energyLossPerStep, energyDepositedVector
):
    nParticles = segmentCounts.shape[0]

    for i in prange(nParticles): 
        count = segmentCounts[i]
        if count == 0:
            continue

        totalLen = 0.0
        for j in range(count):
            totalLen += segmentLengths[i, j]
        if totalLen <= 0.0:
            continue

        inverseTotalLength = 1.0 / totalLen
        for j in range(count):
            frac = segmentLengths[i, j] * inverseTotalLength
            deposit = frac * energyLossPerStep[i]
            x, y, z = voxelIndices[i, j]
            energyDepositedVector[x, y, z] += deposit
            # print(f'Depositing energy: {deposit} at voxel ({x}, {y}, {z})')


def scatteringStepMaterial(
    initialPositions,
    directions,
    initialEnergies,
    energyGrid,
    grid,
    gridShape,
    physicalSize,
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
    
    # Get material indices, segment lengths, segment counts, and voxel indices
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
    
    # Create a mask that is True for valid segments, False otherwise
    validMask = np.arange(maxSegments) < segmentCounts[:, None] 
    firstMaterial = materialIndices[:, 0][:, None]  

    # Check if all valid materials are equal to the first material
    allSame = np.all((materialIndices == firstMaterial) | ~validMask, axis=1)

    # Particles that crossed multiple materials have False in allSame
    crossedMultipleMaterials = ~allSame
    stayedSingleMaterial = allSame
    # print('Crossed multiple materials:', crossedMultipleMaterials)
    # print('Stayed single material:', stayedSingleMaterial)
    
    # Divide particles into those that stayed in a single material and those that crossed multiple
    # If stayed in single material, the stepLength is used completely = 1. 
    if np.any(stayedSingleMaterial):
        singleIndice = np.where(stayedSingleMaterial)[0]
        # print('Stayed single material indices:', singleIndice)
        
        # Sample angles and energies randomly
        IndexMaterials = materialIndices[singleIndice, 0]
        # print('Index materials:', IndexMaterials)
        anglesSample = np.random.uniform(0, 2 * np.pi, size=singleIndice.shape[0])
        energiesSample = initialEnergies[singleIndice] - np.random.rand(singleIndice.shape[0])
        
        # Update the postions of the particles
        newPositions[singleIndice] += stepLength * directions[singleIndice]
        newPositions[singleIndice] = np.round(newPositions[singleIndice], decimals=4) 
        newDirections[singleIndice] = updateDirection(directions[singleIndice], anglesSample)
        
        # Update energy deposition
        energyLossStep = initialEnergies[singleIndice] - energiesSample
        depositEnergy3DStepTraversal(
            voxelIndices=voxelIndices[singleIndice],
            segmentLengths=segmentLengths[singleIndice],
            segmentCounts=segmentCounts[singleIndice],
            energyLossPerStep=energyLossStep,
            energyDepositedVector=energyGrid
        )
        
        # Update the energies of the particles
        newEnergies[singleIndice] = energiesSample
        
    if np.any(crossedMultipleMaterials):
        multipleIndice = np.where(crossedMultipleMaterials)[0]
        # print('Crossed multiple material indices:', multipleIndice)
        
        # Get material indices and segment lengths for multiple materials
        materialsMulti = materialIndices[multipleIndice]
        segmentLengthsMulti = segmentLengths[multipleIndice]
        
        firstMaterial = materialsMulti[:, 0][:, None] 
        # print('First material:', firstMaterial)
        
        # Create mask: True where material equals firstMaterial, False otherwise
        sameAsFirst = (materialsMulti == firstMaterial)
        lengthInFirstMaterial = np.sum(segmentLengthsMulti * sameAsFirst, axis=1)
        # print('Length in first material:', lengthInFirstMaterial)
        
        # Sample angles and energies randomly
        anglesSample = np.random.uniform(0, 2 * np.pi, size=multipleIndice.shape[0])
        energiesSample = initialEnergies[multipleIndice] - np.random.rand(multipleIndice.shape[0])
        # print('Energies sampled:', energiesSample)
        
        energyLossStep = initialEnergies[multipleIndice] - energiesSample
        fractionStep = lengthInFirstMaterial / stepLength
        realEnergyLossStep = energyLossStep * fractionStep
        
        finalEnergy = initialEnergies[multipleIndice] - realEnergyLossStep
        # print('Final energies:', finalEnergy)
        
        # Update the postions of the particles
        newPositions[multipleIndice] += lengthInFirstMaterial[:, None] * directions[multipleIndice]
        newPositions[multipleIndice] = np.round(newPositions[multipleIndice], decimals=4)
        newDirections[multipleIndice] = updateDirection(directions[multipleIndice], anglesSample)
        
        # Truncate segmentLengths, voxelIndices, and segmentCounts for actual path
        segmentLengths_subset = segmentLengths[multipleIndice] 
        voxelIndices_subset = voxelIndices[multipleIndice]   

        truncatedSegmentLengths = np.zeros_like(segmentLengths_subset)
        truncatedVoxelIndices = np.full_like(voxelIndices_subset, -1)

        truncatedSegmentCounts = np.count_nonzero(sameAsFirst, axis=1)
        truncatedSegmentLengths[sameAsFirst] = segmentLengths_subset[sameAsFirst]
        truncatedVoxelIndices[sameAsFirst] = voxelIndices_subset[sameAsFirst]
        # print('Truncated segment lengths:', truncatedSegmentLengths)
        # print('Truncated voxel indices:', truncatedVoxelIndices)
        # print('Truncated segment counts:', truncatedSegmentCounts)

        # Update energy deposition
        depositEnergy3DStepTraversal(
            voxelIndices=truncatedVoxelIndices,
            segmentLengths=truncatedSegmentLengths,
            segmentCounts=truncatedSegmentCounts,
            energyLossPerStep=realEnergyLossStep,
            energyDepositedVector=energyGrid
        )
        
        # Update the energies of the particles
        newEnergies[multipleIndice] = finalEnergy

    return newPositions, newDirections, newEnergies

# Example Usage (demonstrates how to call and what to expect)
if __name__ == "__main__":
    # np.random.seed(41)  # Fix seed for reproducibility
    numberOfParticles = 10000  # Number of particles
    gridShape_tuple = (6, 6, 6) # Use tuple for initial definition
    physicalSize_tuple = (3, 3, 3)  # Half-length in each axis

    # Ensure gridShape and physicalSize are NumPy arrays for Numba-compiled functions
    gridShape_np = np.array(gridShape_tuple, dtype=np.int32)
    physicalSize_np = np.array(physicalSize_tuple, dtype=np.float64)
    energyGrid = np.zeros(gridShape_np, dtype=np.float64)  # Example energy grid, initialized to zero
    # print(energyGrid)
    numMaterials = 4
    stepLength = 1.0

    grid = np.random.randint(0, numMaterials, size=gridShape_np)

    # Initial positions: All particles start at (0, 0, -physicalSize[2])
    initialPositions = np.zeros((numberOfParticles, 3), dtype=np.float64)
    initialPositions[:, 2] = -physicalSize_np[2] + 0.3
    # print("Initial positions:\n", initialPositions)
    
    # Directions: Random, always pointing to +Z
    directions = np.random.uniform(-0.5, 0.5, (numberOfParticles, 3)).astype(np.float64)
    directions[:, 2] = np.random.uniform(0.6, 0.8, numberOfParticles) # Ensure +Z component
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # print("Directions:\n", directions)
    
    initialEnergies = np.ones(numberOfParticles) * 200.0 # Example initial energies

    print('-------------------------------------------------')
    # --- Test dda3dStepTraversal ---
    newPositions, newDirections, newEnergies = scatteringStepMaterial(
        initialPositions=initialPositions,
        directions=directions,
        initialEnergies=initialEnergies,
        energyGrid=energyGrid,
        grid=grid,
        gridShape=gridShape_np,
        physicalSize=physicalSize_np,
        stepLength=stepLength
    )
    
    # print('New positions:\n', newPositions)
    # print('New directions:\n', newDirections)
    # print('New energies:\n', newEnergies)