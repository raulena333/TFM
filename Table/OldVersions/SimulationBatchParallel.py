import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import time
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, shared_memory
import os
import csv

samplerCache = {}

# python3 -m cProfile -o profile_results.prof ReadTableAndSimulateParallel.py
# snakeviz profile_results.prof
# gprof2dot -f pstats profile_results.prof | dot -Tpng -o profile.png

# Plot styling
params = {
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'legend.fontsize': 14
}
pylab.rcParams.update(params) 

class HistogramSampler:
    def __init__(self, hist, angleRange, energyRange):
        self.hist = hist
        self.angleRange = angleRange
        self.energyRange = energyRange
        self.angleBins, self.energyBins = hist.shape

        self.flatHist = hist.flatten()
        self.cumsum = np.cumsum(self.flatHist)
        self.cumsum /= self.cumsum[-1]  # Normalize

        self.angleEdges = np.linspace(angleRange[0], angleRange[1], self.angleBins + 1)
        self.energyEdges = np.linspace(energyRange[0], energyRange[1], self.energyBins + 1)

        self.angleStep = self.angleEdges[1] - self.angleEdges[0]
        self.energyStep = self.energyEdges[1] - self.energyEdges[0]

    def sample(self, size=1):
        randValues = np.random.rand(size)
        idxs = np.searchsorted(self.cumsum, randValues, side='right')
        angleIdxs, energyIdxs = np.unravel_index(idxs, self.hist.shape)

        angles = self.angleEdges[angleIdxs] + 0.5 * self.angleStep
        energies = self.energyEdges[energyIdxs] + 0.5 * self.energyStep

        return angles, energies

def reverseVariableChange(initialEnergy, angle, energy):
    """
    Reverse the variable change applied to the energy loss value and angle.

    Parameters:
    - energy (float): initial energy.

    Returns:
    - realEnergy (float): energy after reversing the variable change.
    - realAngle (float): angle after reversing the variable change.
    """
    # Reverse the angle change
    realAngle = angle / np.sqrt(initialEnergy)  
    # Reverse the energy change
    realEnergy = initialEnergy * (1 - np.exp(energy * np.sqrt(initialEnergy)))  
    
    return realAngle, realEnergy

def variableChange(energy, angle, energyloss):
    """
    Apply the variable change to the energy loss value and angle.
    
    Parameters:
    - energy (float): initial energy.
    - angle (float): angle value.
    - energyloss (float): energy loss value.
    
    Returns:
    - energyChange (float): energy loss value after applying the variable change.
    - angleChange (float): angle value after applying the variable change.
    """
    # Apply the variable change
    energyChange = np.log((energy - energyloss) / energy) / np.sqrt(energy)
    angleChange = angle * np.sqrt(energy) 
    
    return energyChange, angleChange

def sampleReverseCalculate_vectorized(data, material, energies, angleRange, energyRange, materialToIndex):
    materialIdx = materialToIndex[material]
    availableEnergies = data['energies']
    
    roundedEnergies = np.round(energies, 1)
    closestIndices = np.array([
        np.argmin(np.abs(availableEnergies - E)) for E in roundedEnergies
    ])

    sampledAngles = np.zeros_like(energies)
    sampledEnergies = np.zeros_like(energies)

    for idx in np.unique(closestIndices):
        cacheKey = (materialIdx, idx)
        if cacheKey not in samplerCache:
            hist = data['prob_table'][materialIdx, idx]
            samplerCache[cacheKey] = HistogramSampler(hist, angleRange, energyRange)

        sampler = samplerCache[cacheKey]

        indices = np.where(closestIndices == idx)[0]
        angles, energies_ = sampler.sample(size=indices.size)
        realAngles, realEnergies = reverseVariableChange(energies[indices], angles, energies_)
        
        sampledAngles[indices] = realAngles
        sampledEnergies[indices] = realEnergies

    return sampledAngles, sampledEnergies

def createPhysicalSpace(bigVoxel, voxelShapeBins, dt=1 / 3):
    # Create a range of coordinates from -bigVoxel to +bigVoxel with 200 bins
    xRange = np.linspace(-bigVoxel[0] / dt, bigVoxel[0] / dt, voxelShapeBins[0]) 
    yRange = np.linspace(-bigVoxel[1] / dt, bigVoxel[1] / dt, voxelShapeBins[1])
    zRange = np.linspace(-bigVoxel[2] / dt, bigVoxel[2] / dt, voxelShapeBins[2])
    
    return xRange, yRange, zRange

def sampleReverseCalculateInterpolation(data, material, energy, angleRange, energyRange, materialToIndex):
    probTable = data['prob_table']
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
    realAngle, realEnergy = reverseVariableChange(energy, angleSample, energySample)

    return realAngle, realEnergy

def calculateEnergyDepositBin(position, physicalSize, energyLoss, energyDepositedVector, voxelShapeBins):
    """
    Maps energy deposition to a 3D voxel grid with non-cubic dimensions.

    Parameters:
    - position: np.array([x, y, z]) in mm
    - physicalSize: np.array([size_x, size_y, size_z]) in mm — half-lengths along each axis
    - energyLoss: float, energy lost at that point
    - energyDepositedVector: 3D numpy array (voxel grid)
    - voxelShapeBins: tuple, e.g. (50, 50, 250)
    """
    physicalSize = np.squeeze(physicalSize)
    
    # Normalize position from [-size, +size] to [0, 1] along each axis
    normalized = (np.array(position) + physicalSize) / (2 * physicalSize)

    voxelIndices = np.floor(normalized * np.array(voxelShapeBins)).astype(int)
    voxelIndices = np.clip(voxelIndices, [0, 0, 0], np.array(voxelShapeBins) - 1)

    x, y, z = voxelIndices
    energyDepositedVector[x, y, z] += energyLoss

    return energyDepositedVector


def simulateBatchParticles_vectorized(
    batchSize, data, material, initialEnergy,
    angleRange, energyRange, materialToIndex,
    interp, bigVoxelSize, energyDepositedVector, 
    energyThreshold, kVal
):
    energy = np.full(batchSize, initialEnergy)
    position = np.tile([0.0, 0.0, -bigVoxelSize[2]], (batchSize, 1))
    velocity = np.tile([0.0, 0.0, 1.0], (batchSize, 1))
    dt = 1 / 3
    # maxValue = 1
    
    # Move particles a tiny bit forward before the loop
    smallStep = 1e-3  # 0.001 mm or 1 micron, adjust if needed
    position += velocity * smallStep

    active = np.ones(batchSize, dtype=bool)
    angleChanges = np.zeros(batchSize)
    energyChanges = np.zeros(batchSize)

    while np.any(active):
        energyActive = energy[active]
        numberActive = energyActive.size
        
        previousEnergy = energyActive.copy()

        if interp:
            realAngles = np.zeros(numberActive)
            realEnergies = np.zeros(numberActive)
            for i, E in enumerate(energyActive):
                realAngles[i], realEnergies[i] = sampleReverseCalculateInterpolation(
                    data, material, E, angleRange, energyRange, materialToIndex
                )
        else:
            realAngles = np.zeros(numberActive)
            realEnergies = np.zeros(numberActive)
            realAngles, realEnergies = sampleReverseCalculate_vectorized(
                data, material, energyActive, angleRange, energyRange, materialToIndex
            )

        dEnergies, dAngles = variableChange(initialEnergy, realAngles, realEnergies)
        energyLossPerStep = previousEnergy - realEnergies

        # Initialize k, dtScaled, and scaled energy loss
        # k = np.ones(numberActive)             # default: no scaling
        dtScaled = np.ones(numberActive) * dt
        energyLossPerStepScaled = energyLossPerStep.copy()

        # Masks for low and high energy
        # lowEnergyMask = energyActive < energyThreshold
        # highEnergyMask = ~lowEnergyMask

        # Apply scaling only to low-energy particles
        # k[lowEnergyMask] = kVal + (maxValue - kVal) * (energyActive[lowEnergyMask] / energyThreshold)  # between (kVal and maxValue)
        # k[highEnergyMask] = 0.5 + np.random.rand(sum(highEnergyMask))

        # Apply dt and energy scaling based on k
        # dtScaled[lowEnergyMask] *= k[lowEnergyMask]
        # energyLossPerStepScaled[lowEnergyMask] = energyLossPerStep[lowEnergyMask] * k[lowEnergyMask]
        # dtScaled[highEnergyMask] *= k[highEnergyMask]
        # energyLossPerStepScaled[highEnergyMask] = energyLossPerStep[highEnergyMask] * k[highEnergyMask]

        angleChanges[active] = dAngles
        energyChanges[active] = dEnergies
        energy[active] = realEnergies

        phi = np.random.uniform(0, 2 * np.pi, numberActive)
        theta = np.radians(realAngles)

        v = velocity[active]
        perp = np.cross(v, np.array([0, 0, 1]))
        norm = np.linalg.norm(perp, axis=1, keepdims=True)
        perp[norm[:, 0] < 1e-8] = np.array([1, 0, 0])
        perp = perp / np.where(norm == 0, 1, norm)

        cross_perp_v = np.cross(perp, v)
        w = (
            np.cos(theta)[:, np.newaxis] * v +
            np.sin(theta)[:, np.newaxis] * (
                np.cos(phi)[:, np.newaxis] * cross_perp_v +
                np.sin(phi)[:, np.newaxis] * perp
            )
        )
        velocity[active] = w / np.linalg.norm(w, axis=1, keepdims=True)
        
        previousPosition = position[active].copy()
        position[active] += velocity[active] * dtScaled[:, np.newaxis]

        for i, _ in enumerate(np.where(active)[0]):
            calculateEnergyDepositBin(
                previousPosition[i], bigVoxelSize, energyLossPerStepScaled[i],
                energyDepositedVector, energyDepositedVector.shape
            )

        insideVoxel = np.all(
            (position[active] >= -np.array(bigVoxelSize)) &
            (position[active] <= np.array(bigVoxelSize)),
            axis=1
        )
        activeIndex = np.where(active)[0]
        active[activeIndex] = (realEnergies > 0) & insideVoxel

    finalAngles = np.degrees(np.arccos(velocity[:, 2])) * np.sqrt(initialEnergy)
    return finalAngles.tolist(), energyChanges.tolist()
    

def simulateBatchParticles_worker(args):
    (
        shm_name, shape, dtype_str,
        shm_energy_deposited_name, energy_deposited_shape, energy_deposited_dtype_str,
        batchSize, materials, energies,
        material, initialEnergy,
        angleRange, energyRange, materialToIndex,
        interp, bigVoxelSize, energyThreshold,
        kVal
    ) = args

    # Reseed RNG to ensure different results across workers
    np.random.seed(int(time.time() * 1000) % 2**32 + os.getpid())
    
    # Attach to shared memory for prob_table
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    prob_table = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=existing_shm.buf)

    # Attach to shared memory for energyDepositedVector
    existing_shm_energy_deposited = shared_memory.SharedMemory(name=shm_energy_deposited_name)
    energyDepositedVector = np.ndarray(energy_deposited_shape, dtype=np.dtype(energy_deposited_dtype_str), buffer=existing_shm_energy_deposited.buf)

    # Reconstruct the data dictionary in each worker
    data = {
        'prob_table': prob_table,
        'materials': materials,
        'energies': energies
    }

    return simulateBatchParticles_vectorized(
        batchSize, data, material, initialEnergy,
        angleRange, energyRange, materialToIndex,
        interp, bigVoxelSize, energyDepositedVector, 
        energyThreshold, kVal
    )

    
def runMultiprocessedBatchedSim(
    totalSamples, batchSize, numWorkers,
    shm_prob_table, prob_table_shape, prob_table_dtype,
    shm_energy_deposited, energy_deposited_shape, energy_deposited_dtype,
    data, material, initialEnergy,
    angleRange, energyRange, materialToIndex,
    interp, bigVoxelSize, energyThreshold,
    kVal,
):
    numBatches = (totalSamples + batchSize - 1) // batchSize

    argsList = [
        (
            shm_prob_table.name, prob_table_shape, prob_table_dtype.name,
            shm_energy_deposited.name, energy_deposited_shape, energy_deposited_dtype.name,
            min(batchSize, totalSamples - i * batchSize),
            data['materials'], data['energies'], 
            material, initialEnergy,
            angleRange, energyRange, materialToIndex,
            interp, bigVoxelSize, energyThreshold,
            kVal
        )
        for i in range(numBatches)
    ]

    angleDistribution = []
    energyDistribution = []

    with Pool(processes=numWorkers) as pool:
        with tqdm(total=totalSamples, desc="Simulating (batched)") as pbar:
            for result in pool.imap_unordered(simulateBatchParticles_worker, argsList):
                angles, energies = result
                angleDistribution.extend(angles)
                energyDistribution.extend(energies)
                pbar.update(len(angles)) 

    return angleDistribution, energyDistribution

# --- Main Execution ---
if __name__ == "__main__":
    startTime = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--interp", action="store_true", help="Enable interpolation")
    args = parser.parse_args()
    # logging.basicConfig(filename="simulation_errors.log", level=logging.ERROR)

    nProtonsTable = 1000000
    npzPath = f'./Table/4DTable{nProtonsTable}.npz'
    samplingN = 1000000
    numberOfBins = 100 
    material = 'G4_WATER'
    initialEnergy = 200.0
    savePath = './PlotsSimulation/'
    npyPath = './Numpy/'
    csvPath = './CSV/'
    timingPath = './Timing/'
    angleRange = (0, 70)
    energyRange = (-0.57, 0)
    bigVoxelSize = (33.3333, 33.33333, 50)
    voxelShapeBins = (50, 50, 300)
    dt = 1 / 3

    # Make sure the timing directory exists
    # os.makedirs(timingPath, exist_ok=True)
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(npyPath, exist_ok=True)
    os.makedirs(csvPath, exist_ok=True)
    
    rawData = np.load(npzPath, allow_pickle=True)

    # Fully extract arrays here
    prob_table = rawData['prob_table']
    materials = rawData['materials']
    energies = rawData['energies']
    
    # Create shared memory for the table
    shm_prob_table = shared_memory.SharedMemory(create=True, size=prob_table.nbytes)
    shm_prob_np = np.ndarray(prob_table.shape, dtype=prob_table.dtype, buffer=shm_prob_table.buf)
    np.copyto(shm_prob_np, prob_table)
    
    # Store in plain dict — safe for multiprocessing
    data = {
        'prob_table': np.array(prob_table),
        'materials': list(materials),
        'energies': np.array(energies)
    }
    rawData.close()

    materialToIndex = {mat: idx for idx, mat in enumerate(data['materials'])}
    
    # Create shared memory for the energy deposited
    energyDeposited = np.zeros(voxelShapeBins, dtype=np.float32)
    shm_energy_deposited = shared_memory.SharedMemory(create=True, size=energyDeposited.nbytes)
    shm_energy_deposited_np = np.ndarray(energyDeposited.shape, dtype=energyDeposited.dtype, buffer=shm_energy_deposited.buf)
    np.copyto(shm_energy_deposited_np, energyDeposited)
    
    angleDistribution = []
    energyDistribution = []
    
    timeOfHistories = []
    
    # Custom k sampling
    energyThresh = 15.
    k = 0.9
    
    try:
        print(f"Running simulation with energy threshold: {energyThresh}, k: {k}")
        shm_energy_deposited_np.fill(0)

        batchSize = 1000
        numWorkers = cpu_count()

        prob_table_shape = prob_table.shape
        prob_table_dtype = prob_table.dtype

        angleDistribution, energyDistribution = runMultiprocessedBatchedSim(
                    samplingN, batchSize, numWorkers,
                    shm_prob_table, prob_table_shape, prob_table_dtype,
                    shm_energy_deposited, energyDeposited.shape, energyDeposited.dtype,
                    data, material, initialEnergy,
                    angleRange, energyRange, materialToIndex,
                    args.interp, bigVoxelSize, energyThresh, k
        )
                
        energyVector3D = np.ndarray(energyDeposited.shape, dtype=energyDeposited.dtype, buffer=shm_energy_deposited.buf).copy()
        # x, y, z = np.nonzero(energyVector3D)
        # energies = energyVector3D[x, y, z]

        # energyGrid = np.zeros(voxelShapeBins)

        # for xi, yi, zi, ei in zip(x, y, z, energies):
        #     energyGrid[xi, yi, zi] = ei

        # projectionXZ = np.sum(energyGrid, axis=1)  # axis=1 is Y
        # np.save(f'{npyPath}projectionXZSimulation.npy', energyVector3D)

        # # Create coordinate ranges
        # xRange, _, zRange = createPhysicalSpace(bigVoxelSize, voxelShapeBins)

        # # Plot the projection
        # fig, ax = plt.subplots(figsize=(8, 6))
        # im = ax.imshow(
        #             projectionXZ.T,
        #             extent=[xRange[0], xRange[-1], zRange[0], zRange[-1]],
        #             origin='lower',
        #             aspect='auto',
        #             cmap='Blues'
        # )
        # ax.axhline(y=110, color='red', linestyle='--', linewidth=1.5, label='Z = 100 mm')

        # cbar = plt.colorbar(im, ax=ax)
        # cbar.set_label('Summed Energy Deposit (MeV)', fontsize=12)
        # cbar.ax.tick_params(labelsize=10)

        # ax.set_xlabel('X (mm)')
        # ax.set_ylabel('Z (mm)')

        # plt.tight_layout()
        # plt.savefig(f"{savePath}EnergyDeposit_XZ_ProjectionSimulation.pdf", dpi=300)
        # plt.close(fig)
                
        # # Profile of the beam at X = 0 and X-Axis
        # indxCenter = energyVector3D.shape[0] // 2
        # profileZ = energyVector3D[indxCenter, :, :]
        # profileZMean = profileZ.mean(axis=0)
                
        # zIndex = 50
        # profileX = energyVector3D[:, :, zIndex]
        # profileXMean = profileX.mean(axis=1)

        # fig1, ax1 =plt.subplots(1, 2, figsize=(10, 6))
        # ax1[0].plot(zRange, profileZMean)
        # ax1[0].set_xlabel(r'Z voxel Index')
        # ax1[0].set_ylabel(r'Energy Deposit (MeV)')
        # # ax1[0].set_xlim(- bigVoxelSize[2] / dt, + bigVoxelSize[2] / dt)
                
        # ax1[1].plot(xRange, profileXMean)
        # ax1[1].set_xlabel(r'X voxel Index')
        # ax1[1].set_ylabel(r'Energy Deposit (MeV)')
        # # ax1[1].set_xlim(-bigVoxelSize[0] / dt, + bigVoxelSize[0] / dt)
                
        # plt.tight_layout()
        # plt.savefig(f'{savePath}ProfilesEnergyDepositSimulation.pdf')
        # plt.close(fig1)

    #     fileforEnergyDeposit = f"{csvPath}EnergyAtBoxByBinsMySimulation.csv"
    #     with open(fileforEnergyDeposit, 'w', newline='') as file:
    #         # Write the header manually with #
    #         file.write(
    #                 "# Simulation Version: 4.\n"
    #                 "# Results for scorer: EnergyDeposit\n"
    #                 "# Scored in component: Box\n"
    #                 "# EnergyDeposit (MeV): Sum\n"
    #                 f"# X in {voxelShapeBins[0]} bins of {200 / voxelShapeBins[0]} mm\n"
    #                 f"# Y in {voxelShapeBins[1]} bins of {200 / voxelShapeBins[1]} mm\n"
    #                 f"# Z in {voxelShapeBins[2]} bins of {200 / voxelShapeBins[2]} mm\n"
    #             )
    #         writer = csv.writer(file, delimiter=' ')
                    
    #         # Write voxel data line by line
    #         for x in range(energyVector3D.shape[0]):
    #             for y in range(energyVector3D.shape[1]):
    #                 for z in range(energyVector3D.shape[2]):
    #                     value = energyVector3D[x, y, z]
    #                     if value > 0:
    #                         writer.writerow([x, y, z, f"{value:.6f}"])
                                    
    #         totalEnergy = energyVector3D.sum()
    #         writer.writerow([f"Sum : {totalEnergy:.6f}"])
                
    # # Cleanup shared memory     
    finally:
        shm_prob_table.close()
        shm_prob_table.unlink()
        shm_energy_deposited.close()
        shm_energy_deposited.unlink()
        
    endTime = time.time()
    print(f"Simulation time: {endTime - startTime:.2f} seconds")
     
    # Plot
    # plt.plot(samplingN, timeOfHistories, linestyle='-',marker = '.', color="black")      
    # plt.xlabel("Number of Histories")
    # plt.ylabel("Time (s)")
    # plt.tight_layout()  
    # plt.savefig(f"{savePath}HistoryTimePlotForEnergiesBatch.pdf")
    # # plt.show()
    # plt.close()

    # Plot histograms
    # fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    # sns.histplot(energyDistribution, bins=numberOfBins, edgecolor="black", color='orange', kde=False, ax=axs[0]) 
    # axs[0].set_xlabel(r'$\frac{ln((E_i-E_f)/E_i)}{\sqrt{E_i}}$ (MeV$^{-1/2}$)')
    # axs[0].set_title('Final Energy distribution')
    # axs[0].set_yscale('log')
            
    # sns.histplot(angleDistribution, bins=numberOfBins, edgecolor="black", color='red', kde=False, ax=axs[1])
    # axs[1].set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
    # axs[1].set_title('Final Angles distribution')
    # axs[1].set_yscale('log')
    
    # plt.tight_layout()
    # plt.savefig(f'{savePath}MySimulationHistogramsBatchSharedVectorizedVector.pdf')
    # plt.close(fig)


    # hist1, xedges1, yedges1 = np.histogram2d(angleDistribution, energyDistribution, bins=numberOfBins)
    # finalProbabilities = hist1 / np.sum(hist1)

    # fig2, axs2 = plt.subplots(figsize=(8, 6))
    # h1 = axs2.pcolormesh(xedges1, yedges1, finalProbabilities.T, cmap='Reds', shading='auto')
    # fig2.colorbar(h1, ax=axs2, label='Probability')
    # axs2.set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
    # axs2.set_ylabel(r'$ln((E_i-E_f)/E_i)\sqrt{E_i}$ (MeV$^{-1/2}$)')

    # plt.tight_layout()
    # plt.savefig(f'{savePath}2DMySimulationHistogramsBatch.pdf')
    # plt.close(fig2) 