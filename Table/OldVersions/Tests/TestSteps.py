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

@numba.jit(nopython=True, fastmath=True)
def dda3dNextStep(pos, dir, gridShape, physicalSize):
    # Normalize the direction vector for robustness in this standalone function
    dir_magnitude = np.linalg.norm(dir)
    if dir_magnitude == 0:
        return np.full(3, -1, dtype=np.int32), np.inf
    dir_norm = dir / dir_magnitude

    # Calculate voxel size for the full physical extent (2 * physicalSize)
    full_physical_extent = 2 * physicalSize
    voxel_size = full_physical_extent / gridShape

    # Current voxel coordinates (integer indices) using the helper function
    ix, iy, iz = getVoxelIndex(pos, physicalSize, gridShape)
    current_voxel = np.array([ix, iy, iz], dtype=np.int32)

    # Initialize tMax (distance to next grid line intersection along each axis)
    # and tDelta (distance to cross one full voxel along each axis)
    tMax = np.zeros(3, dtype=np.float64)
    tDelta = np.zeros(3, dtype=np.float64)
    step = np.zeros(3, dtype=np.int32)

    # Shift position from [-physicalSize, physicalSize] range to [0, full_physical_extent] range
    pos_shifted = pos + physicalSize

    for i in range(3):
        if dir_norm[i] >= 0:
            step[i] = 1
            # Distance to the next positive grid line
            # Use ceil to ensure we're looking at the *next* grid line if exactly on one.
            next_grid_line = (current_voxel[i] + 1) * voxel_size[i]
            tMax[i] = next_grid_line - pos_shifted[i]
        else: # dir_norm[i] < 0
            step[i] = -1
            # Distance to the previous negative grid line
            # Use floor to ensure we're looking at the *previous* grid line if exactly on one.
            next_grid_line = current_voxel[i] * voxel_size[i]
            tMax[i] = pos_shifted[i] - next_grid_line

        # Handle the case where tMax is very small due to floating point.
        # If particle is effectively on the boundary, tMax might be ~0 or slightly negative.
        # We want to treat it as if it's already past the boundary for the purpose of this step.
        # A small epsilon added to tMax can prevent picking up the *same* boundary.
        if tMax[i] < 1e-9: # If very close to or past the boundary
            tMax[i] = voxel_size[i] # Treat as if we're now at the start of the next voxel, distance is full voxel size
            # This will make the particle step *into* the next voxel, effectively
            # preventing an infinite loop of zero-length steps at a boundary.

        # Avoid division by zero for components of dir_norm that are zero
        if abs(dir_norm[i]) > 1e-20:
            tMax[i] /= abs(dir_norm[i])
            tDelta[i] = voxel_size[i] / abs(dir_norm[i])
        else:
            tMax[i] = np.inf
            tDelta[i] = np.inf

    min_t_index = np.argmin(tMax)
    length = tMax[min_t_index]

    next_voxel = current_voxel.copy()
    next_voxel[min_t_index] += step[min_t_index]

    if not (next_voxel[0] >= 0 and next_voxel[0] < gridShape[0] and \
            next_voxel[1] >= 0 and next_voxel[1] < gridShape[1] and \
            next_voxel[2] >= 0 and next_voxel[2] < gridShape[2]):
        return np.full(3, -1, dtype=np.int32), np.inf

    return next_voxel, length

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

def scatteringUpdateMaterial(
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
    n = initialPositions.shape[0]
    updatePositions = initialPositions.copy()
    updateDirections = directions.copy()
    finalEnergies = np.zeros_like(initialEnergies)

    # Get voxel indices and segment lengths using the JIT-compiled dda3dStepTraversal
    voxelIndices, segmentLengths, segmentCounts = dda3dStepTraversal(
        initialPositions, directions, stepLength, grid, gridShape, physicalSize, maxSegments
    )
    print(f"VoxelIndices: {voxelIndices}")
    print(f"SegmentCounts: {segmentCounts}")

    segmentIndices = np.arange(maxSegments)
    mask = segmentIndices < segmentCounts[:, None]  # shape (n, maxSegments)

    # Flatten arrays and apply mask
    # Only consider valid entries (not the -1 padding)
    valid_voxelIndices_flat = voxelIndices[mask]
    valid_segmentLengths_flat = segmentLengths[mask]

    # Get materials index for valid voxel indices
    # This requires numpy's advanced indexing, which works outside of nopython mode
    # Ensure gridShape is passed as numpy array to grid and getVoxelIndex
    materialsIndex = grid[
        valid_voxelIndices_flat[:, 0],
        valid_voxelIndices_flat[:, 1],
        valid_voxelIndices_flat[:, 2]
    ]
    print(f"MaterialsIndex: {materialsIndex}")

    # Split valid arrays per particle
    splitIndices = np.cumsum(segmentCounts)[:-1] if segmentCounts.size > 1 else np.array([], dtype=np.int32)
    materialsPerParticle = np.split(materialsIndex, splitIndices)
    print(f"MaterialsPerParticle: {materialsPerParticle}")
    segmentsPerParticle = np.split(valid_segmentLengths_flat, splitIndices)

    # Sanity check
    assert all(len(m) == len(s) for m, s in zip(materialsPerParticle, segmentsPerParticle)), \
        "Mismatch in per-particle segment lengths!"

    # Identify which particles stayed in the same material
    sameMaterialMask = np.array([
        len(np.unique(materials)) == 1 for materials in materialsPerParticle
    ])

    print(f"[{', '.join(map(str, np.where(sameMaterialMask)[0]))}] particles have same material")
    print(f"[{', '.join(map(str, np.where(~sameMaterialMask)[0]))}] particles changed material")

    # --- Vectorized: All in same material ---
    if np.any(sameMaterialMask):
        idxSame = np.where(sameMaterialMask)[0]
        
        materials_for_idxSame = np.array([
        grid[voxelIndices[p_idx, 0, 0], voxelIndices[p_idx, 0, 1], voxelIndices[p_idx, 0, 2]]
        for p_idx in idxSame
        ])
        print(f"Len of materials_for_idxSame: {len(materials_for_idxSame)}")
        print(f'Length of initialEnergies[idxSame]: {len(initialEnergies[idxSame])}')
        
        thetaSamples = np.random.uniform(0, np.pi, len(idxSame))
        energySamples = (initialEnergies[idxSame] - np.random.rand(len(idxSame)))
        newPositions, newDirections = updatePosition(
            initialPositions[idxSame],
            directions[idxSame],
            stepLength, 
            thetaSamples
        )
        finalEnergies[idxSame] = energySamples
        updatePositions[idxSame] = newPositions
        updateDirections[idxSame] = newDirections
        energyLossStep = initialEnergies[idxSame] - energySamples
        
        depositEnergy3DStepTraversal(
            voxelIndices[idxSame],
            segmentLengths[idxSame],
            segmentCounts[idxSame],
            energyLossStep,
            energyGrid
        )
        
    # --- Loop: Particles that changed material ---
    if np.any(~sameMaterialMask):
        idxChange = np.where(~sameMaterialMask)[0]
        for i_global in idxChange:
            pos = initialPositions[i_global].copy() # Make copy to avoid modifying original
            dir = directions[i_global].copy() # Make copy

            length_accum = 0.0

            current_voxel_tuple = getVoxelIndex(pos, physicalSize, gridShape)
            current_material = grid[current_voxel_tuple[0], current_voxel_tuple[1], current_voxel_tuple[2]]
            current_energy = initialEnergies[i_global]
            
            # print('-----------------------------------------')
            # print(f"Particle {i_global}: Initial position {pos}, Initial direction {dir}, "
            #       f"Initial voxel {current_voxel_tuple}, Initial material {current_material}")

            finalEnergies[i_global] = initialEnergies[i_global]  # Use initial energy for this particle
            while length_accum < stepLength:
                next_voxel, travel_len_suggested = dda3dNextStep(pos, dir, gridShape, physicalSize)
                # print(f'Result of next step: ({next_voxel}, {travel_len_suggested})')

                # Check for out of bounds using the sentinel value or inf length
                if travel_len_suggested == np.inf:
                    # Move remaining distance along current direction
                    # (assuming particle exits and doesn't scatter further)
                    pos += dir * (stepLength - length_accum)
                    length_accum = stepLength # Mark as done
                    break

                # The `next_voxel` from dda3dNextStep is the voxel the ray *enters*.
                # If `travel_len_suggested` is less than `stepLength - length_accum`,
                # the particle will enter a new voxel.
                # If `travel_len_suggested` is equal to or greater, it finishes the step
                # within the current (or last entered) voxel.

                remaining_step_length = stepLength - length_accum
                actual_travel_len = min(travel_len_suggested, remaining_step_length)

                # Move particle to the point of scattering/voxel boundary
                pos += dir * actual_travel_len
                length_accum += actual_travel_len

                # Get material at the new current position (which might be the *next* voxel if a boundary was crossed)
                current_voxel_tuple = getVoxelIndex(pos, physicalSize, gridShape)
                # Ensure current_voxel_tuple is within bounds before accessing grid
                if not (current_voxel_tuple[0] >= 0 and current_voxel_tuple[0] < gridShape[0] and
                        current_voxel_tuple[1] >= 0 and current_voxel_tuple[1] < gridShape[1] and
                        current_voxel_tuple[2] >= 0 and current_voxel_tuple[2] < gridShape[2]):
                    # Particle exited the grid in this sub-step
                    break
                current_material = grid[current_voxel_tuple[0], current_voxel_tuple[1], current_voxel_tuple[2]]
                print(f"Particle {i_global}: Current position {pos}, "
                      f"Current voxel {current_voxel_tuple}, Current material {current_material}, ")

                # Apply scattering *after* moving to the next voxel boundary / within current material, and energy deposition
                # The 'realAngles' should be derived from the material's properties for scattering
                # For this example, we'll use a random angle, but this is where your physics model comes in.
                theta_scatter = np.random.uniform(0, 20) # Example scattering angle
                
                energy_sampled = current_energy - np.random.rand()
                energy_weight = energy_sampled / stepLength  # Weight the energy by the step length, divinding by stepLength
                loss_voxel = (current_energy - energy_weight)
                current_energy = energy_weight
                
                # Deposit energy loss in the energy grid
                energyGrid[current_voxel_tuple[0], current_voxel_tuple[1], current_voxel_tuple[2]] += loss_voxel
                
                # updatePosition takes single particle data (pos[None], dir[None]) and returns single particle
                _, dir_new_scattered = updatePosition(pos[None], dir[None], 0.0, np.array([theta_scatter])) # Scatter in place
                dir = dir_new_scattered[0] # Update direction for next sub-step

                print(f"Scattering: New position {pos}, New direction {dir}, "
                      f"Accumulated length {length_accum:.4f}, Current material {current_material}, "
                      f"Travel length {actual_travel_len:.4f}")

                if abs(length_accum - stepLength) < 1e-9: # Check if entire stepLength is covered
                    break
                
            finalEnergies[i_global] = current_energy
            updatePositions[i_global] = pos
            updateDirections[i_global] = dir

    return updatePositions, updateDirections, finalEnergies


# Example Usage (demonstrates how to call and what to expect)
if __name__ == "__main__":
    # np.random.seed(45)  # Fix seed for reproducibility
    numberOfParticles = 5  # Number of particles
    gridShape_tuple = (2, 2, 2) # Use tuple for initial definition
    physicalSize_tuple = (1, 1, 1)  # Half-length in each axis

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
    # print("Initial positions:\n", initialPositions)

    # Directions: Random, always pointing to +Z
    directions = np.random.uniform(-1, 1, (numberOfParticles, 3)).astype(np.float64)
    directions[:, 2] = np.random.uniform(0.1, 1.0, numberOfParticles) # Ensure +Z component
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # print("Directions:\n", directions)
    
    initialEnergies = np.ones(numberOfParticles) * 200.0 # Example initial energies

    # --- Test dda3dStepTraversal ---
    voxelIndices, segmentLengths, segmentCounts = dda3dStepTraversal(
        initialPositions, directions, stepLength, grid, gridShape_np, physicalSize_np
    )

    # --- Test scatteringUpdateMaterial ---
    updatePositions, updateDirections, finalEnergies= scatteringUpdateMaterial(
        initialPositions,
        directions,
        initialEnergies,
        energyGrid,
        grid,
        gridShape_np,
        physicalSize_np,
        stepLength,
        maxSegments=5
    )
    