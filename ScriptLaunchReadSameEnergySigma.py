import re
import os
import time
import struct
import h5py
import numpy as np
from scipy.stats import skew, kurtosis
import subprocess
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

params = {
    'xtick.labelsize': 12,    
    'ytick.labelsize': 12,      
    'axes.titlesize' : 12,
    'axes.labelsize' : 12,
    'legend.fontsize': 12,
}
pylab.rcParams.update(params)  # Apply changes

def modifyBeamEnergy(filePath, newEnergy):
    """
    Modify the beam energy in a TOPAS input file and save it with the same name.

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - newEnergy (float): New energy value to replace in the file.
    """
    patternEnergy = r"(dv:So/MySource/BeamEnergySpectrumValues = 1 )(\d+(\.\d+)?)( MeV)"

    with open(filePath, 'r') as file:
        content = file.read()
    
    # Replace the energy value in the matched line
    updatedFile = re.sub(patternEnergy, rf"dv:So/MySource/BeamEnergySpectrumValues = 1 {newEnergy} MeV", content)
    
    with open(filePath, 'w') as file:
        file.write(updatedFile)

    print(f"Updated beam energy to {newEnergy} MeV in {filePath}.")
    
def modifyInputParameters(filePath, numberOfRuns):
    """
    Modify input parameters, such as the number of runs

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - numberOfRuns (int): Define the number of protons simulated (Number of histories in run).
    """
    # Regular expression pattern to match the line with the number of runs
    pattern = r"(i:So/MySource/NumberOfHistoriesInRun = )(\d+)"
    
    with open(filePath, 'r') as file:
        content = file.read()

    # Replace the number of runs in the matched line
    updatedFile = re.sub(pattern, rf"i:So/MySource/NumberOfHistoriesInRun = {numberOfRuns}", content)
    
    with open(filePath, 'w') as file:
        file.write(updatedFile)

    print(f"Updated number of runs to {numberOfRuns} in {filePath}.")
    
def modifySeedRandomness(filePath, seed):
    """
    Modify input parameters, such as the number of runs

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - seed (int): Define the randomness of the run.
    """
    # Regular expression pattern to match the line with the number of runs
    pattern = r"(i:Ts/Seed = )(\d+)"
    
    with open(filePath, 'r') as file:
        content = file.read()

    # Replace the number of runs in the matched line
    updatedFile = re.sub(pattern, rf"i:Ts/Seed = {seed}", content)
    
    with open(filePath, 'w') as file:
        file.write(updatedFile)

    print(f"Updated seed number in {filePath}.")
    
    
def runTopas(filePath, dataPath):
    """
    Run TOPAS txt-script with the modified input file

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - dataPath (str): Path to the TOPAS G4 data.
    """
    try:
        result = subprocess.run(f'export TOPAS_G4_DATA_DIR={dataPath} && ~/topas/bin/topas {filePath}', 
                        text=True, shell=True)
        if result == 0:
            print("Data loaded and simulation have started succesfully ")
            
    except FileNotFoundError:
        print("TOPAS executable not found. Make sure TOPAS is installed and in your PATH.")

        
def trueRandomNumber(minValue = 0, maxValue = 100):
    """Random number in the interval [minVal, maxVal]."""
    randBytes = os.urandom(4)
    randInt = struct.unpack("I", randBytes)[0]
    return minValue + (randInt % (maxValue - minValue + 1)) 
        
        
def readInputFileSigma(fileName, initialEnergy):
    newData = np.loadtxt(fileName)  
    print(f'{fileName} loaded successfully.')
    finalDirectionCosineX, finalDirectionCosineY, finalEnergy, isSign = newData[:, [3,4,5,8]].T
    
    # logEnergy = np.log(initialEnergy - finalEnergy / initialEnergy)
    # Calculate statistics, mean, variance(“Bessel’s correction”), sigma, median
    meanEnergy = np.mean(finalEnergy)
    varianceEnergy = np.var(finalEnergy, ddof=1)
    stdEnergy = np.std(finalEnergy, ddof=1)
    medianEnergy = np.median(finalEnergy)
        
    # Percentiles (IQR, 90), etc... are values that divide a dataset into 100 equal parts. Each percentile tells you the value below which a certain percentage of the data falls
    # IQR=Q3-Q1
    # q1Energy = np.percentile(finalEnergy, 25)
    # q3Energy = np.percentile(finalEnergy, 75)
    # iqr = q3Energy - q1Energy

    # Skew and kurtosis
    # skewnessEnergy = skew(finalEnergy) 
    # kurtEnergy = kurtosis(finalEnergy)  
        
    finalAngles = []
    # Loop over each particle's direction cosines
    for directionX, directionY, sign in zip(finalDirectionCosineX, finalDirectionCosineY, isSign):
        directionZ = np.sqrt(1 - directionX**2 - directionY**2)
            
        # Adjust sign of directionZ based on isSign
        if sign == 0:
            directionZ *= -1
        angle = np.degrees(np.arccos(directionZ))
        finalAngles.append(angle)
            
    # Calculate statistics, mean, variance(“Bessel’s correction”), sigma, median
    meanAngle = np.mean(finalAngles)
    varianceAngle = np.var(finalAngles, ddof=1)
    stdAngle = np.std(finalAngles, ddof=1)
    medianAngle = np.median(finalAngles)
        
    return finalEnergy, finalAngles, meanEnergy, varianceEnergy, stdEnergy, medianEnergy, meanAngle, varianceAngle, stdAngle, medianAngle


def saveH5File(h5FileName, dictionaryData):
    """
    Saves statistical data as columns under an incrementing group name (Simulation 1, Simulation 2, etc.)
    
    Parameters:
    - hdf5FileName (str): Path to the HDF5 file.
    - All the other lists contain statistical values to be saved.
    """  
    
    with h5py.File(h5FileName, "a") as f:
        # Find the simulation number
        simNumber = 1
        while f"Simulation{simNumber}" in f:
            simNumber += 1
        group = f.create_group(f"Simulation{simNumber}")
        
        # Save each array as a column dataset inside the group
        for key, data in dictionaryData.items():
            group.create_dataset(key, data=data)
            

if __name__ == "__main__":
    
    start_time = time.time()  # Record the start time
    
    # Variables
    numberOfProtonsRun = 1000
    numberOfRuns = 1000000
    energy = 100
    
    numberOfBinsAngles = 100
    numberOfBinsEnergies = 100
    jointNumberOfBins = 100
    
    dataPath = '~/G4Data/'
    voxelPhaseFile = "./MyVoxelPhaseSpace.txt"
    fileName = "OutputVoxel.phsp"
    outputH5File = "SimulationMeanStd.h5"
    
    # Variables to store statistics for each run
    meanEnergies = []
    varianceEnergies = []
    stdEnergies = []
    medianEnergies = []
    
    meanAngles = []
    varianceAngles = []
    stdAngles = []
    medianAngles = []
    
    # Data to save in a dictionary
    dataDict = {}
    
    # Call the function to change input parameters
    modifyInputParameters(voxelPhaseFile, numberOfProtonsRun)
    modifyBeamEnergy(voxelPhaseFile, energy)
    
    # Process each run and store the statistics
    for i in range(numberOfRuns):
        
        # Change seed randomness
        # modifySeedRandomness(voxelPhaseFile, trueRandomNumber(0, 2147483647))
        modifySeedRandomness(voxelPhaseFile, i + 1) # starts from 1
        
        # Simulate the experiment
        runTopas(voxelPhaseFile, dataPath)
        
        # Read simulation output and calculate statistics
        finalEnergy, finalAngles, meanEnergy, varianceEnergy, stdEnergy, medianEnergy, meanAngle, varianceAngle, stdAngle, medianAngle = readInputFileSigma(fileName, energy)

        if meanEnergy is None:
            print(f"Skipping run {i} due to error or missing data.")
            continue

        # Append the statistics to the respective lists
        meanEnergies.append(meanEnergy)
        varianceEnergies.append(varianceEnergy)
        stdEnergies.append(stdEnergy)
        medianEnergies.append(medianEnergy)
        
        meanAngles.append(meanAngle)
        varianceAngles.append(varianceAngle)
        stdAngles.append(stdAngle)
        medianAngles.append(medianAngle)
    
    # Fill the dictionary 
    dataDict = {
            "meanEnergy": np.array(meanEnergies),
            "varianceEnergy": np.array(varianceEnergies),
            "stdEnergy": np.array(stdEnergies),
            "medianEnergy": np.array(medianEnergies),
            "meanAngle": np.array(meanAngles),
            "varianceAngle": np.array(varianceAngles),
            "stdAngle": np.array(stdAngles),
            "medianAngle": np.array(medianAngles),
        }
    
    saveH5File(fileName, dataDict)
          
    # Plot histograms of energy and angle
    fig1, axs1 = plt.subplots(1, 2, figsize=(10, 6))

    sns.histplot(meanEnergies, bins=numberOfBinsEnergies, edgecolor="black", color='orange', kde=False, ax=axs1[0])
    axs1[0].set_xlabel(r'$E[E_f]$ (MeV)')
    axs1[0].set_title('Final mean Energy distribution')
    axs1[0].set_yscale('log')
            
    sns.histplot(stdEnergies, bins=numberOfBinsEnergies, edgecolor="black", color='red', kde=False, ax=axs1[1])
    axs1[1].set_xlabel(r'$\sigma_{E_f}$ (deg)')
    axs1[1].set_title('Final Std Energy distribution')
    axs1[1].set_yscale('log')

    plt.tight_layout()
    # plt.show()
    savefileName = f'HistogramStdMeanEnergy.pdf'
    plt.savefig(savefileName)
    plt.close(fig1) 
    
    
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 6))
    
    sns.histplot(meanAngles, bins=numberOfBinsAngles, edgecolor="black", color='orange', kde=False, ax=axs2[0])
    axs2[0].set_xlabel(r'$E[\theta]$ (MeV)')
    axs2[0].set_title('Final Mean Angle distribution')
    axs2[0].set_yscale('log')
            
    sns.histplot(stdAngles, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False, ax=axs2[1])
    axs2[1].set_xlabel(r'$\sigma_\theta$ (deg)')
    axs2[1].set_title('Final Std Angle distribution')
    axs2[1].set_yscale('log')

    plt.tight_layout()
    # plt.show()
    savefileName = f'HistogramStdMeanAngle.pdf'
    plt.savefig(savefileName)
    plt.close(fig2) 
    
    
    # Compute 2D histogram https://matplotlib.org/stable/gallery/mplot3d/surface3d.html
    hist1, xMean, yMean = np.histogram2d(meanAngles, meanEnergies, bins=jointNumberOfBins)
    hist2, xStd, yStd = np.histogram2d(stdAngles, stdEnergies, bins=jointNumberOfBins)
    probabilitiesMean = hist1 / np.sum(hist1)
    probabilitiesStd = hist2 / np.sum(hist2)

    # Create grid for plotting
    Xmean, Ymean = np.meshgrid(xMean[:-1], yMean[:-1])
    Zmean = probabilitiesMean.T
    Xstd, Ystd = np.meshgrid(xStd[:-1], yStd[:-1])
    Zstd = probabilitiesStd.T

    # 3D Plot and surface mean
    fig3 = plt.figure(figsize=(10, 7))
    ax1 = fig3.add_subplot(111, projection='3d')
    surf = ax1.plot_surface(Xmean, Ymean, Zmean, cmap='coolwarm', edgecolor='k', alpha=0.9)
    ax1.set_xlabel(r'$E[\theta]$ (deg)')
    ax1.set_ylabel(r'$E[E_f]$ (MeV)')
    # ax1.set_zlabel(r'Probability')
    # ax.set_title('Joint Probability Distribution')

    cbar = plt.colorbar(surf, ax=ax1, shrink=0.5, pad=0.05, aspect = 4)
    cbar.set_label('Probability')
    plt.tight_layout()
    plt.savefig('ProbabilityDistributionMean.pdf')
    # plt.show()
    plt.close(fig3) 
    
    # 3D Plot and surface std
    fig4 = plt.figure(figsize=(10, 7))
    ax2 = fig4.add_subplot(111, projection='3d')
    surf = ax2.plot_surface(Xstd, Ystd, Zstd, cmap='coolwarm', edgecolor='k', alpha=0.9)
    ax2.set_xlabel(r'$\sigma_\theta$ (deg)')
    ax2.set_ylabel(r'$\sigma_{E_f}$ (MeV)')
    # ax2.set_zlabel(r'Probability')
    # ax.set_title('Joint Probability Distribution')

    cbar = plt.colorbar(surf, ax=ax2, shrink=0.5, pad=0.05, aspect = 4)
    cbar.set_label('Probability')
    plt.tight_layout()
    plt.savefig('ProbabilityDistributionStd.pdf')
    # plt.show()
    plt.close(fig4) 
    
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the difference

    print(f"Program finished in {execution_time:.2f} seconds.")