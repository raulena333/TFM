import numpy as np
import numba

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
def dda3dNextStep(pos, dir, gridShape, physicalSize):
    dirMag = np.linalg.norm(dir)
    if dirMag == 0.0:
        return np.array([-1, -1, -1], dtype=np.int32), np.inf  # Invalid direction

    # Normalize direction
    dirNorm = dir / dirMag
    voxelSize = (2.0 * physicalSize) / gridShape
    posShifted = pos + physicalSize

    # Compute current voxel index
    currentVoxel = np.array(getVoxelIndex(pos, physicalSize, gridShape), dtype=np.int32)
    # Initialize arrays
    tMax = np.empty(3, dtype=np.float64)
    step = np.empty(3, dtype=np.int32)

    for i in range(3):
        if dirNorm[i] > 0.0:
            step[i] = 1
            nextBoundary = (currentVoxel[i] + 1) * voxelSize[i]
            tMax[i] = (nextBoundary - posShifted[i]) / dirNorm[i]
        elif dirNorm[i] < 0.0:
            step[i] = -1
            prevBoundary = currentVoxel[i] * voxelSize[i]
            tMax[i] = (posShifted[i] - prevBoundary) / -dirNorm[i]
        else:
            step[i] = 0
            tMax[i] = np.inf

        if tMax[i] < 1e-6:
            tMax[i] = voxelSize[i] / max(abs(dirNorm[i]), 1e-12)

    length = np.min(tMax)

    return currentVoxel, length

# @numba.jit(nopython=True, fastmath=True)
def dda3dNextStepBatch(
    positionsBatch, directionsBatch, gridShape, physicalSize,
    travelLengthsOutput, nextVoxelsOutput
):
    numParticles = positionsBatch.shape[0]
    
    for i in range(numParticles):
        pos = positionsBatch[i]
        dir = directionsBatch[i]

        # Call the existing scalar dda3dNextStep for each particle
        nextVoxelsSuggested, travelLenSuggested = dda3dNextStep(pos, dir, gridShape, physicalSize)
        
        travelLengthsOutput[i] = travelLenSuggested
        nextVoxelsOutput[i] = nextVoxelsSuggested

def scatteringStepMaterial(
    initialPositions,
    directions,
    initialEnergies,
    energyGrid,
    grid,
    gridShape,
    physicalSize,
    stepLength=1.0,
):
    numberOfParticles = initialPositions.shape[0]
    stepLengthAccumulated = np.zeros(numberOfParticles, dtype=np.float32)
    active = np.ones(numberOfParticles, dtype=bool)
    
    while np.any(active):
        # Active particles
        positionsBatch, directionsBatch, energyBatch = initialPositions[active], directions[active], initialEnergies[active]
                
        # Allocate output arrays for DDA
        ddaLengths = np.zeros(positionsBatch.shape[0], dtype=np.float32)
        nextVoxels = np.zeros((positionsBatch.shape[0], 3), dtype=np.int32)

        dda3dNextStepBatch(positionsBatch, directionsBatch, gridShape, physicalSize, ddaLengths, nextVoxels)
        print('DDA Lengths: ', ddaLengths)
        print('Current Voxels: ', nextVoxels)
        
        # Clamp travel lengths to the maximum number of segments, do not exceed stepLength   
        remainingStepLengths = stepLength - stepLengthAccumulated[active]
        clampedLengths = np.minimum(remainingStepLengths, ddaLengths)
        print('Clamped Lengths: ', clampedLengths)

        # Update positions and energies based on the next voxels
        realAngles = np.random.rand(clampedLengths.shape[0]) 
        realEnergies = energyBatch - np.random.rand(clampedLengths.shape[0])
        newPositions = positionsBatch + clampedLengths * directionsBatch
        roundedPositions = np.round(newPositions, decimals=6)
        newDirections = updateDirection(directionsBatch, realAngles)
        print('New Positions: ', newPositions)
        print('Rounded Positions: ', roundedPositions)
        print('New Directions: ', newDirections)
        # Get materialIndex at the end of position
        midpoint = 0.5 * (positionsBatch + newPositions)
        materialIndices = getMaterialIndexAtPosition(midpoint, grid, physicalSize, energyGrid.shape)
        # print('Material Indices: ', materialIndices)
        # print('Voxel midpoint: ', getVoxelIndex(midpoint[0], physicalSize, energyGrid.shape))
        
        # Update input arrays in-place
        initialPositions[active] = roundedPositions
        directions[active] = newDirections
        initialEnergies[active] = realEnergies
        stepLengthAccumulated[active] += clampedLengths
        
        # Set inactive if stepLength has exceed stepLength
        active[stepLengthAccumulated >= stepLength] = False
        
    return initialPositions, directions, initialEnergies
        

# Example Usage (demonstrates how to call and what to expect)
if __name__ == "__main__":
    # np.random.seed(41)  # Fix seed for reproducibility
    numberOfParticles = 1  # Number of particles
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
    print(grid)

    # Initial positions: All particles start at (0, 0, -physicalSize[2])
    initialPositions = np.zeros((numberOfParticles, 3), dtype=np.float64)
    initialPositions[:, 2] = -physicalSize_np[2]
    print("Initial positions:\n", initialPositions)
    print("Initial voxels in grid:\n", getVoxelIndex(initialPositions[0], physicalSize_np, gridShape_np))
    print('Material index', getMaterialIndexAtPosition(initialPositions, grid, physicalSize_np, gridShape_np))
    
    # Directions: Random, always pointing to +Z
    directions = np.random.uniform(-0.5, 0.5, (numberOfParticles, 3)).astype(np.float64)
    directions[:, 2] = np.random.uniform(0.6, 0.8, numberOfParticles) # Ensure +Z component
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    print("Directions:\n", directions)
    
    initialEnergies = np.ones(numberOfParticles) * 200.0 # Example initial energies

    print('-------------------------------------------------')
    # --- Test dda3dStepTraversal ---
    for i in range(4):
        curPositions, curDirections, curEnergies = scatteringStepMaterial(
            initialPositions,
            directions,
            initialEnergies,
            energyGrid,
            grid,
            gridShape_np,
            physicalSize_np,
            stepLength
        )