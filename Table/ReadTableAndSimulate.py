import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import os
import time
import argparse

params = {
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,      
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'legend.fontsize': 14
}
pylab.rcParams.update(params)  # Apply changes

def sampleReverseCalculate(data, material, energy, angleRange, energyRange):
    """
    Sample (angle, energy) values from a 2D histogram and reverse the variable change.
    
    Parameters:
    - data (dict): Dictionary containing the histogram data.
    - material (str): Name of the material.
    - energy (float): Energy value.
    
    Returns:
    - realAngle (float): Real angle value.
    - realEnergy (float): Real energy value.
    """

    # Extract metadata
    probTable = data['prob_table']
    materials = data['materials']
    energies = data['energies']

    # Validate and locate material and energy
    if material not in materials:
        raise ValueError(f"Material '{material}' not found in the dataset.")
    if energy not in energies:
        energy = np.round(energy, 1)
        raise ValueError(f"Energy '{energy}' not found in the dataset.")

    materialIdx = np.where(materials == material)[0][0]
    energyIdx = np.where(energies == energy)[0][0]

    # Retrieve the histogram for the specified material and energy
    hist = probTable[materialIdx, energyIdx]
    
    # Sample one (angle, energy) pair
    angleSample, energySample = sampleFromHist(hist, angleRange, energyRange) 
    # Reverse the variable change
    realAngle, realEnergy = reverseVariableChange(energy, angleSample, energySample)  
    
    return realAngle, realEnergy

def sampleReverseCalculateInterpolation(data, material, energy, angleRange, energyRange):
    # Extract metadata
    probTable = data['prob_table']
    energies = data['energies']
    materials = data['materials']
    
    energies = np.sort(energies)

    # Get material index
    if material not in materials:
        raise ValueError(f"Material '{material}' not found in data.")
    materialIdx = np.where(materials == material)[0][0]
    
    # Find surrounding energy indices
    if energy < energies[0] or energy > energies[-1]:
        raise ValueError(f"Energy {energy} out of bounds ({energies[0]} - {energies[-1]})")

    lowerIndex = np.searchsorted(energies, energy) -1
    upperIndex = lowerIndex + 1
    
    # Clamp to valid range
    lowerIndex = max(0, lowerIndex)
    upperIndex = min(len(energies) - 1, upperIndex)
    
    energyLow = energies[lowerIndex]
    energyUp = energies[upperIndex]
    
    probLow = probTable[materialIdx, lowerIndex]
    probHigh = probTable[materialIdx, upperIndex]
    
    # It computes a weighted average of the two histograms based on how close energy is to each:
    if energyUp == energyLow:
        interpolatedHist = probLow
    else:
        weight = (energy - energyLow) / (energyUp - energyLow)
        interpolatedHist = (1 - weight) * probLow + weight * probHigh
        
    # Sample one (angle, energy) pair
    angleSample, energySample = sampleFromHist(interpolatedHist, angleRange, energyRange) 
    # Reverse the variable change
    realAngle, realEnergy = reverseVariableChange(energy, angleSample, energySample)  
    
    return realAngle, realEnergy

def sampleFromHist(hist, angleRange, energyRange):
    """
    Sample (angle, energy) values from a 2D histogram.

    Parameters:
    - hist (2D np.array): Normalized joint probability histogram.
    - angleRange (tuple): Min/max range of transformed angles.
    - energyRange (tuple): Min/max range of transformed energy losses.

    Returns:
    - angle (float): Sampled angle (transformed).
    - energy (float): Sampled energy (transformed).
    """
    hist = hist / np.sum(hist)  # Normalize the histogram

    # Flatten and sample using probabilities
    flatHist = hist.flatten()
    idx = np.random.choice(len(flatHist), p=flatHist)

    # Convert back to 2D index
    angleBins, energyBins = hist.shape
    angleIdx, energyIdx = np.unravel_index(idx, (angleBins, energyBins))

    # Convert indices to actual values (bin centers)
    angleStep = (angleRange[1] - angleRange[0]) / angleBins
    energyStep = (energyRange[1] - energyRange[0]) / energyBins

    angle = angleRange[0] + (angleIdx + 0.5) * angleStep
    energy = energyRange[0] + (energyIdx + 0.5) * energyStep

    return angle, energy

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

def intersectionVecWithPlane(origin, direction, planePoint, planeNormal):
    # Normalize the ray direction
    direction = direction / np.linalg.norm(direction)

    # Calculate the dot product between the plane normal and the ray direction
    denominator = 0
    denominator = np.dot(planeNormal, direction)
    
    # If the denominator is 0, the ray is parallel to the plane, no intersection
    if np.abs(denominator) < 1e-6:
        return None, None

    # Calculate the intersection parameter t
    intermediate = planePoint - origin
    t = np.dot(intermediate, planeNormal) / denominator

    # If t >= 0, the intersection is in front of the ray's origin
    if t >= 0:
        intersection_point = origin + t * direction
        intersection_point = np.round(intersection_point, 3)
        
        return intersection_point, planeNormal
    else:
        return None, None  # No valid intersection, it's behind the ray

def plotSampledDistribution(hist, numSamples, angleRange=(0, 70), energyRange=(-0.57, 0)):
    # Generate samples
    angles = []
    energies = []
    
    for h in range(numSamples):
        print(f"Sampling {h+1} ")
        angle, energy = sampleFromHist(hist, angleRange, energyRange)
        angles.append(angle)
        energies.append(energy)

    # Alternatively, we can also plot a 2D histogram of the samples
    plt.figure(figsize=(8, 6))
    plt.hist2d(angles, energies, bins=70, range=[angleRange, energyRange], cmap='Blues')
    plt.colorbar(label='Frequency')
    plt.xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
    plt.ylabel(r'$\frac{ln((E_i-E_f)/E_i)}{\sqrt{E_i}}$ (MeV$^{-1/2}$)')
    plt.savefig('SampledDistribution.pdf')
    plt.close()
    

def checkIfInsideBigVoxel(position, voxelSize):
    x, y, z = position
    return (
        -voxelSize <= x <= voxelSize and
        -voxelSize <= y <= voxelSize and
        -voxelSize <= z <= voxelSize
    )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Choose simulation mode, interpolation and debugging.")
    parser.add_argument("--interp", action="store_true", help="Use interpolation between energy table values (default: False).")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for sampling tests and extra output.")
    args = parser.parse_args()

    nProtonsTable = 100000
    npzPath = f'./Table/4DTable{nProtonsTable}.npz' 
    samplingN = 100
    material = 'G4_WATER'
    initialEnergy = 200.0
    sameMaterial = True  # Have we changed the material?
    numberOfBins = 100
    savePath = './MySimulation/'
    
    # Define the ranges used in the histogram transformation
    angleRange = (0, 70)  # degrees * sqrt(E_i)
    energyRange = (-0.57, 0)  # log10((E_i - E_f)/E_i) / sqrt(E_i)

    bigVoxelSize = 1e3 / 2 # (mm) maximum size for the simulation voxel
    
    normals = np.array([
        [0, 0, 1],     # +Z face
        [0, 0, -1],    # -Z face
        [1, 0, 0],     # +X face
        [-1, 0, 0],    # -X face
        [0, 1, 0],     # +Y face
        [0, -1, 0]     # -Y face
    ])

    centers = np.array([
        [0, 0, 0.5],    # +Z
        [0, 0, -0.5],   # -Z
        [0.5, 0, 0],    # +X
        [-0.5, 0, 0],   # -X
        [0, 0.5, 0],    # +Y
        [0, -0.5, 0]    # -Y
    ])

    # Load the whole table
    data = np.load(npzPath, allow_pickle=True)

    # Plot the sampled distribution
    # plotSampledDistribution(hist, samplingN)
    
    # Save the histogram data for the specified material and energy
    angleDistribution = []
    energyDistribution = []
    
    # Calculate time of simulation
    starTime = time.time()
    
    # Start the simulation
    for l in range(samplingN):
        if args.debug:
            print(f"Testing example {l+1} of {samplingN}")
        # Initial position (x, y, z)
        position = np.array([0, 0, -bigVoxelSize], dtype=float) # Current position
        positionVoxel = np.array([0,0, -bigVoxelSize], dtype=int) # Current voxel position
        
        energyChange, angleChange = 0, 0
        boolBigVox = True
        energy = initialEnergy
        
        while energy > 0 and checkIfInsideBigVoxel(position, bigVoxelSize):
            # Sample (angle, energy) values from the histogram
            if args.interp:
                realAngle, realEnergy = sampleReverseCalculateInterpolation(data, material, energy, angleRange, energyRange)
            else:
                realAngle, realEnergy = sampleReverseCalculate(data, material, energy, angleRange, energyRange)
            
            # Apply the variable change to the energy loss value and angle
            energyChange, angleChange = variableChange(energy, realAngle, realEnergy)
            if args.debug:
                print(f"Energy: {realEnergy}, Angle : {realAngle}\n")
            energy = realEnergy

            phi = np.random.uniform(0, 2 * np.pi) 
            # Calculate the new position based on the angle and step size, for now we assume a simple straight line motion
            directionX = np.sin(np.radians(realAngle)) * np.cos(phi)
            directionY = np.sin(np.radians(realAngle)) * np.sin(phi)
            directionZ = np.cos(np.radians(realAngle))
            
            directionVector = np.array([directionX, directionY, directionZ])
            
            foundIntersection = False
            intersection = None
            intersectedNormal = []
        
            for i in range(len(normals)):
                intersection, normal = intersectionVecWithPlane(position, directionVector, centers[i], normals[i])
    
                if intersection is not None:
                    intersection = np.round(intersection, 6)  # Clean small floats
                    intersectedNormal = normal
                    # if args.debug:
                    #     print(f"Intersection at {intersection} with normal {normal}")
                    foundIntersection = True
                    break
                
            if not foundIntersection:
                if args.debug:
                    print("No valid intersection found. Exiting loop.")
                break  # Prevent infinite loops
                
            position = intersection + 1e-6 * directionVector  # Small nudge forward
            
            # Update voxel index
            delta = np.array(intersectedNormal, dtype=int)
            positionVoxel += delta
            
            if args.debug:
                print(f"Position : {position}, Voxel : {positionVoxel}\n")

            # Handle material switch if necessary (assuming sameMaterial is a function or condition)
            if sameMaterial:
                continue
            
        if args.debug:
            print(f"Final Angle: {angleChange}, energy: {energyChange}")
        angleDistribution.append(angleChange)
        energyDistribution.append(energyChange)
        
    endTime = time.time()
    elapsedTime = endTime - starTime
    print(f"Simulation time: {elapsedTime:.2f} seconds")
    
    os.makedirs(savePath, exist_ok=True) # Create directory if it doesn't exist
    
    # Plot the angle and energy distributions
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    sns.histplot(energyDistribution, bins=numberOfBins, edgecolor="black", color='orange', kde=False, ax=axs[0])
    axs[0].set_xlabel(r'$\frac{\ln((E_i - E_f)/E_i)}{\sqrt{E_i}}$ \ (MeV$^{-1/2}$)')
    axs[0].set_title('Final Energy distribution')
    axs[0].set_yscale('log')
         
    sns.histplot(angleDistribution, bins=numberOfBins, edgecolor="black", color='red', kde=False, ax=axs[1])
    axs[1].set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
    axs[1].set_title('Final Angles distribution')
    axs[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{savePath}OutputHistograms.pdf')
    plt.close(fig)

    # Compute 2D Histogram
    hist1, xedges1, yedges1 = np.histogram2d(angleDistribution, energyDistribution, bins=numberOfBins, range = (angleRange, energyRange))
    finalProbabilities = hist1 / np.sum(hist1)

    fig2, axs2 = plt.subplots(figsize=(8, 6))
    h1 = axs2.pcolormesh(xedges1, yedges1, finalProbabilities.T, cmap='Reds', shading='auto')
    fig2.colorbar(h1, ax=axs2, label='Probability')
    axs2.set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
    axs2.set_ylabel(r'$ln((E_i-E_f)/E_i)/ln\sqrt{E_i}$ (ln(MeV)$^{-1}$)')

    plt.tight_layout()
    plt.savefig(f'{savePath}Output2DHistograms.pdf')
    plt.close(fig2)