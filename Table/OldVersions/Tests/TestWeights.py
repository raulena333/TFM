import numpy as np
from numba import prange
import numba
import time

def dda2dPathMaterialFractions(initialPositions, finalPositions, grid, voxelShapeBins, physicalSize, num_materials):
    n = initialPositions.shape[0]
    binsX, binsY = voxelShapeBins
    sizeX, sizeY = physicalSize
    voxelSize = np.array([2 * sizeX / binsX, 2 * sizeY / binsY])
    gridOrigin = np.array([-sizeX, -sizeY])
    materialFractions = np.zeros((n, num_materials), dtype=np.float64)

    for i in range(n):
        p0 = initialPositions[i]
        p1 = finalPositions[i]
        segment = p1 - p0
        total_len = np.linalg.norm(segment)
        if total_len < 1e-12:
            continue
        direction = segment / total_len
        pos = p0.copy()

        # Compute starting voxel indices
        voxel = np.empty(2, dtype=np.int32)
        for j in range(2):
            voxel[j] = int(np.floor((pos[j] - gridOrigin[j]) / voxelSize[j]))

        step = np.empty(2, dtype=np.int32)
        t_max = np.empty(2, dtype=np.float64)
        t_delta = np.empty(2, dtype=np.float64)

        for j in range(2):
            if abs(direction[j]) < 1e-20:
                step[j] = 0
                t_max[j] = 1e30
                t_delta[j] = 1e30
            else:
                if direction[j] > 0:
                    step[j] = 1
                    next_boundary = gridOrigin[j] + (voxel[j] + 1) * voxelSize[j]
                else:
                    step[j] = -1
                    next_boundary = gridOrigin[j] + voxel[j] * voxelSize[j]
                t_max[j] = (next_boundary - pos[j]) / direction[j]
                t_delta[j] = voxelSize[j] / abs(direction[j])

        traveled = 0.0

        while traveled < total_len:
            axis = np.argmin(t_max)
            travel_to_boundary = t_max[axis] - traveled
            travel_len = min(travel_to_boundary, total_len - traveled)

            ix, iy = voxel[0], voxel[1]
            if 0 <= ix < binsX and 0 <= iy < binsY:
                mat = grid[iy, ix]
                if 0 <= mat < num_materials:
                    materialFractions[i, mat] += travel_len

            traveled += travel_len

            if traveled >= total_len:
                break

            voxel[axis] += step[axis]
            t_max[axis] += t_delta[axis]

    # Normalize to fractions
    row_sums = np.sum(materialFractions, axis=1)
    for i in range(n):
        if row_sums[i] > 0:
            materialFractions[i] /= row_sums[i]

    return materialFractions

def dda3dPathMaterialFractions(initialPositions, finalPositions, grid, voxelShapeBins, physicalSize, num_materials):
    n = initialPositions.shape[0]
    binsX, binsY, binsZ = voxelShapeBins
    sizeX, sizeY, sizeZ = physicalSize
    voxelSize = np.array([2 * sizeX / binsX, 2 * sizeY / binsY, 2 * sizeZ / binsZ])
    gridOrigin = np.array([-sizeX, -sizeY, -sizeZ])
    materialFractions = np.zeros((n, num_materials), dtype=np.float64)

    for i in range(n):
        p0 = initialPositions[i]
        p1 = finalPositions[i]
        segment = p1 - p0
        total_len = np.linalg.norm(segment)
        if total_len < 1e-12:
            continue
        direction = segment / total_len
        pos = p0.copy()

        voxel = np.empty(3, dtype=np.int32)
        voxel[0], voxel[1], voxel[2] = getVoxelIndex(pos, physicalSize, voxelShapeBins)

        step = np.empty(3, dtype=np.int32)
        t_max = np.empty(3, dtype=np.float64)
        t_delta = np.empty(3, dtype=np.float64)

        for j in range(3):
            if abs(direction[j]) < 1e-20:
                step[j] = 0
                t_max[j] = 1e30
                t_delta[j] = 1e30
            else:
                if direction[j] > 0:
                    step[j] = 1
                    next_boundary = gridOrigin[j] + (voxel[j] + 1) * voxelSize[j]
                else:
                    step[j] = -1
                    next_boundary = gridOrigin[j] + voxel[j] * voxelSize[j]
                t_max[j] = (next_boundary - pos[j]) / direction[j]
                t_delta[j] = voxelSize[j] / abs(direction[j])

        traveled = 0.0

        while traveled < total_len:
            axis = np.argmin(t_max)
            travel_to_boundary = t_max[axis] - traveled
            travel_len = min(travel_to_boundary, total_len - traveled)

            ix, iy, iz = voxel[0], voxel[1], voxel[2]

            if (0 <= ix < binsX) and (0 <= iy < binsY) and (0 <= iz < binsZ):
                mat = grid[ix, iy, iz]
                if 0 <= mat < num_materials:
                    materialFractions[i, mat] += travel_len

            traveled += travel_len

            if traveled >= total_len:
                break

            voxel[axis] += step[axis]
            t_max[axis] += t_delta[axis]

    # Normalize fractions
    row_sums = np.sum(materialFractions, axis=1)
    for i in range(n):
        if row_sums[i] > 0:
            materialFractions[i] /= row_sums[i]

    return materialFractions

def dda3dVoxelTraversalSegments(initialPositions, finalPositions, grid, energyLosses, voxelShapeBins, physicalSize, max_segments=100):
    n = initialPositions.shape[0]
    binsX, binsY, binsZ = voxelShapeBins
    sizeX, sizeY, sizeZ = physicalSize
    voxelSize = np.array([2 * sizeX / binsX, 2 * sizeY / binsY, 2 * sizeZ / binsZ])
    gridOrigin = np.array([-sizeX, -sizeY, -sizeZ])

    voxelIndices = -np.ones((n, max_segments, 3), dtype=np.int32)
    segmentLengths = np.zeros((n, max_segments), dtype=np.float32)
    segmentCounts = np.zeros(n, dtype=np.int32)

    for i in range(n):
        p0 = initialPositions[i]
        p1 = finalPositions[i]
        dE = energyLosses[i]
        segment = p1 - p0
        total_len = np.sqrt(np.sum(segment ** 2))
        if total_len < 1e-12:
            continue

        direction = segment / total_len
        pos = p0.copy()
        voxel = np.array(getVoxelIndex(pos, physicalSize, voxelShapeBins))

        step = np.empty(3, dtype=np.int32)
        t_max = np.empty(3, dtype=np.float64)
        t_delta = np.empty(3, dtype=np.float64)

        for j in range(3):
            if abs(direction[j]) < 1e-20:
                step[j] = 0
                t_max[j] = 1e30
                t_delta[j] = 1e30
            else:
                if direction[j] > 0:
                    step[j] = 1
                    next_boundary = gridOrigin[j] + (voxel[j] + 1) * voxelSize[j]
                else:
                    step[j] = -1
                    next_boundary = gridOrigin[j] + voxel[j] * voxelSize[j]
                t_max[j] = (next_boundary - pos[j]) / direction[j]
                t_delta[j] = voxelSize[j] / abs(direction[j])

        traveled = 0.0
        count = 0
        energyDeposition = 0.0

        while traveled < total_len and count < max_segments:
            axis = np.argmin(t_max)
            travel_to_boundary = t_max[axis] - traveled
            travel_len = min(travel_to_boundary, total_len - traveled)
            e_portion = dE * (travel_len / total_len)

            ix, iy, iz = voxel
            if 0 <= ix < binsX and 0 <= iy < binsY and 0 <= iz < binsZ:
                material = grid[ix, iy, iz]
                voxelIndices[i, count, 0] = ix
                voxelIndices[i, count, 1] = iy
                voxelIndices[i, count, 2] = iz
                print(f'Travel new segment:', count, 'at voxel:', ix, iy, iz, 'length:', travel_len, 'material:', material)
                segmentLengths[i, count] = travel_len
                count += 1
                energyDeposition += e_portion
                print(f"Energy deposition for segment {count}: {e_portion}, Total energy: {energyDeposition}")
                
                

            traveled += travel_len
            if traveled >= total_len:
                break

            voxel[axis] += step[axis]
            t_max[axis] += t_delta[axis]

        segmentCounts[i] = count

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

def getMaterialIndexAtPosition(positions, grid, physicalSize, voxelShapeBins):
    n = positions.shape[0]
    indices = np.empty(n, dtype=np.int32)
    
    for i in range(n):
        ix, iy, iz = getVoxelIndex(positions[i], physicalSize, voxelShapeBins)
        indices[i] = grid[ix, iy, iz]
        
    return indices

def depositEnergy3DStepTraversal(
    voxelIndices, segmentLengths, segmentCounts, energyLossPerStep,
    # energyDepositedVector, 
    ny, nz
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
            linear = x * ny * nz + y * nz + z
            # energyDepositedVector[linear] += deposit
            print(f"Depositing {deposit} energy at voxel ({x}, {y}, {z}) with linear index {linear}")
        print(f"Total energy deposited for particle {i}: {energyLossPerStep[i]}") 
@numba.jit(nopython=True, fastmath=True)
def dda3dStepTraversal(initialPositions, directions, stepLength, grid, gridShape, physicalSize, max_segments=5):
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

    for i in prange(n):
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

def sampling3dStepTraversal(voxelIndices, segmentLengths, segmentCounts, grid):
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
        print(f'Segment voxels for count {count}:\n', seg_voxels)
        seg_lengths = segmentLengths[indices, :count]  # shape (m, count)
        print(f'Segment lengths for count {count}:\n', seg_lengths)

        # Flatten for efficient material lookup:
        flat_voxels = seg_voxels.reshape(-1, 3)
        #print(f'Flat voxels for count {count}:\n', flat_voxels)

        # Get materials for all those voxels:
        materials = grid[flat_voxels[:, 0], flat_voxels[:, 1], flat_voxels[:, 2]]
        #print(f'Materials for count {count}:\n', materials)

        theta_samples = np.random.uniform(0, np.pi, size=materials.shape[0])
        energy_samples = np.random.uniform(0, 1, size=materials.shape[0])
        #print(f"Sampling for count {count}: theta_samples {theta_samples}, energy_samples {energy_samples}")
        #print(f'Materials shape: {materials.shape}')
        # Reshape back to (m, count)
        theta_samples = theta_samples.reshape(len(indices), count)
        energy_samples = energy_samples.reshape(len(indices), count)
        # print(f'Theta samples for count {count}:\n', theta_samples)
        # print(f'Energy samples for count {count}:\n', energy_samples)

        # Weighted sums:
        weighted_theta = np.sum(theta_samples * seg_lengths, axis=1)
        weighted_energy = np.sum(energy_samples * seg_lengths, axis=1)
        total_len = np.sum(seg_lengths, axis=1)

        # Weighted averages:
        samples[indices] = weighted_theta / total_len
        energy[indices] = weighted_energy / total_len

    return samples, energy

def sampling3dStepTraversal_flat(voxelIndices, segmentLengths, segmentCounts, grid):
    n_particles = voxelIndices.shape[0]
    total_segments = np.sum(segmentCounts)

    # Flatten to only used segments
    valid_indices = np.repeat(np.arange(n_particles), segmentCounts)
    flat_voxels = voxelIndices[valid_indices, np.tile(np.arange(segmentCounts.max()), n_particles)[:total_segments], :]
    flat_lengths = segmentLengths[valid_indices, np.tile(np.arange(segmentCounts.max()), n_particles)[:total_segments]]

    # Get material values
    flat_materials = grid[flat_voxels[:, 0], flat_voxels[:, 1], flat_voxels[:, 2]]

    # Sample theta and energy
    theta_samples = np.random.uniform(0, np.pi, total_segments)
    energy_samples = np.random.uniform(0, 1, total_segments)

    # Weighted sums
    weighted_theta = theta_samples * flat_lengths
    weighted_energy = energy_samples * flat_lengths

    # Accumulate using np.bincount
    samples = np.zeros(n_particles, dtype=np.float64)
    energy = np.zeros(n_particles, dtype=np.float64)
    total_length = np.bincount(valid_indices, weights=flat_lengths, minlength=n_particles)
    sum_theta = np.bincount(valid_indices, weights=weighted_theta, minlength=n_particles)
    sum_energy = np.bincount(valid_indices, weights=weighted_energy, minlength=n_particles)

    nonzero = total_length > 0
    samples[nonzero] = sum_theta[nonzero] / total_length[nonzero]
    energy[nonzero] = sum_energy[nonzero] / total_length[nonzero]

    return samples, energy


@numba.jit(nopython=True, fastmath=True)
def sampling3dStepTraversal_numba(voxelIndices, segmentLengths, segmentCounts, grid_flat, binsX, binsY, binsZ, theta_rand, e_dep_rand):
    n = voxelIndices.shape[0]

    samples = np.zeros(n, dtype=np.float64)
    energy = np.zeros(n, dtype=np.float64)

    idx = 0  # index in pre-generated random arrays

    for i in numba.prange(n):
        count = segmentCounts[i]
        if count == 0:
            samples[i] = 0.0
            energy[i] = 0.0
            continue

        weighted_theta = 0.0
        weighted_energy = 0.0
        total_len = 0.0

        for j in range(count):
            ix = voxelIndices[i, j, 0]
            iy = voxelIndices[i, j, 1]
            iz = voxelIndices[i, j, 2]

            if ix < 0 or ix >= binsX or iy < 0 or iy >= binsY or iz < 0 or iz >= binsZ:
                continue

            length = segmentLengths[i, j]
            total_len += length

            # flatten index for grid lookup
            flat_index = ix + iy * binsX + iz * binsX * binsY
            material = grid_flat[flat_index]

            theta = theta_rand[idx] * np.pi
            e_dep = e_dep_rand[idx]
            idx += 1

            weighted_theta += theta * length
            weighted_energy += e_dep * length

        if total_len > 0:
            samples[i] = weighted_theta / total_len
            energy[i] = weighted_energy / total_len
        else:
            samples[i] = 0.0
            energy[i] = 0.0

    return samples, energy


# === Test the function ===
if __name__ == "__main__":
    np.random.seed(42)  # Fix seed for reproducibility
    numberOfParticles = 2  # Number of particles
    gridShape = (20, 20, 20)
    physicalSize = (10, 10, 10)  # Physical size of the domain (half-length in each axis)
    numMaterials = 5  # Number of 
    stepLength = 1  # Step length for traversal
    
    grid = np.random.randint(0, numMaterials, size=gridShape)  # Random materials from 0 to numMaterials-1pe)
    initialPositions = np.random.uniform(-10, 10, (numberOfParticles, 3))  # One initial position in 2D
    # Clip initial positions to the physical size
    for i in range(3):
        initialPositions[:, i] = np.clip(initialPositions[:, i], -physicalSize[i], physicalSize[i])
    print("Clipped Initial positions:\n", initialPositions)
    
    energyLosses = np.random.uniform(0.1, 1.0, numberOfParticles)  # Random energy losses for each particle
    print("Energy Losses:\n", energyLosses)
    
    directions = np.random.uniform(-1, 1, (numberOfParticles, 3))  # Random directions 
    # Normalize directions to unit length
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    print("Directions:\n", directions)
    
    timeStart= time.time()
    # Compute material fractions using DDA in 3D
    voxelIndices, segmentLengths, segmentCounts = dda3dStepTraversal(
        initialPositions, directions, stepLength, grid, gridShape, physicalSize, max_segments=5
    )
    timeEnd = time.time()
    print(f'DDA took {timeEnd - timeStart:.4f} seconds')
    print('Segment Counts:', segmentCounts)
    print('Voxel Indices:\n', voxelIndices)
    print('Segment Lengths:\n', segmentLengths)
    
    # Time
    timeStart = time.time()
    samples, energy = sampling3dStepTraversal(voxelIndices, segmentLengths, segmentCounts, grid)
    timeEnd = time.time()
    print(f'Sampling vectorized took {timeEnd - timeStart:.4f} seconds')
    
    depositEnergy3DStepTraversal(
        voxelIndices, segmentLengths, segmentCounts, energyLosses,
        # energyDepositedVector, 
        gridShape[1], gridShape[2]
    )
    
    # print('-----------------------------------------')
    # timeStartNumba = time.time()
    # # Pre-generate random samples for theta and energy deposition
    # total_segments = np.sum(segmentCounts)
    # theta_rand = np.random.random(total_segments)
    # e_dep_rand = np.random.random(total_segments)
    # gridFlat = grid.flatten()
    # samples_numba, energy_numba = sampling3dStepTraversal_numba(voxelIndices, segmentLengths, segmentCounts, gridFlat, gridShape[0], gridShape[1], gridShape[2], theta_rand, e_dep_rand)
    # timeEndNumba = time.time()
    # print(f'Sampling numba took {timeEndNumba - timeStartNumba:.4f} seconds')
    
    # print('-----------------------------------------')
    # timeStartFlat = time.time()
    # gridFlat = grid.flatten()
    # samples_flat, energy_flat = sampling3dStepTraversal_numba(voxelIndices, segmentLengths, segmentCounts, gridFlat, gridShape[0], gridShape[1], gridShape[2], theta_rand, e_dep_rand)
    # timeEndFlat = time.time()
    # print(f'Sampling flat took {timeEndFlat - timeStartFlat:.4f} seconds')