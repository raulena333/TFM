import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pylab as pylab
import seaborn as sns
import argparse
import os
import matplotlib.colors as colors

# Matplotlib params
params = {
    'xtick.labelsize': 17,    
    'ytick.labelsize': 17,      
    'axes.titlesize': 17,
    'axes.labelsize': 17,
    'legend.fontsize': 17
}
pylab.rcParams.update(params)  # Apply changes


def returnDoseEnergyValue(energyFile, doseFile):
    
    if not os.path.exists(energyFile):
        print(f"Error: File {energyFile} does not exist.")
        return None, None
    
    if not os.path.exists(doseFile):
        print(f"Error: File {doseFile} does not exist.")
        return None, None

    # Load data, skipping first 5 rows
    dataEnergy = np.loadtxt(energyFile, skiprows=5)
    dataDose = np.loadtxt(doseFile, skiprows=5)
    
    return dataEnergy, dataDose

def calculateEnergyLossDose(energyValues, initialEnergy, mass):

    energyLoss = 0 # MeV
    for energy in energyValues:
        loss = initialEnergy - energy
        energyLoss += loss 
        
    energyJ = 1.6022e-13 * energyLoss # J / Kg
        
    return energyLoss, energyJ / mass


# Energy values
energies = [100, 12]

# Define argument parser
parser = argparse.ArgumentParser(description="Analyze and visualize simulation data with different threshold options.")
parser.add_argument('--uniform', action='store_true',
                    help="Use uniform thresholds for filtering (if not set, uses predefined thresholds by default).")

args = parser.parse_args()

# Path for saving plots
if args.uniform:
    savePath = "./Plots/UniformThreshold/"
else:
    savePath = "./Plots/NoThreshold/"

# Create directory if it doesn't exist
if not os.path.exists(savePath):
    os.makedirs(savePath)
    
# Storage for results
percentajeDiscarded = []

# Number of bins for histograms
numberOfBinsAngles = 100
numberOfBinsEnergies = 100
jointNumberOfBins = 100

# Size of the cube of 1 mm of water
size = 1 # mm
density = 1e-3  # g/mm^3
volume = size * size * size # mm^3
mass = density * volume * 1e-3 # kg

energyDosePath = f'./EnergyDoseFiles/'

energySimulation = []
doseSimulation = []
energyCutOff = []
doseCutOff = []
doseCalculatedSim = []
relativeDifferenceDose = []
relativeDifferenceEnergy = []

# Load data from files
for i, energy in enumerate(energies):   
    fileName = f"OutputVoxel{energy}MeV.phsp"
    energyFile = f"{energyDosePath}EnergyVoxel{energy}MeV.csv"
    doseFile = f"{energyDosePath}DoseVoxel{energy}MeV.csv"
    
    try:
        discardedData = 0
        newData = np.loadtxt(fileName)
        print(f'{fileName} loaded successfully.')

        # Extract relevant columns
        finalDirectionCosineX, finalDirectionCosineY, finalEnergy, isSign, initialEnergy = newData[:, [3,4,5,8,10]].T
        initialData = len(initialEnergy)
        
        if energy == 200:
            mask = finalEnergy > 195
        if energy == 175:
            mask = finalEnergy > 170
        if energy == 150:
            mask = finalEnergy > 145
        if energy == 125:
            mask = finalEnergy > 120
        if energy == 100:
            mask = finalEnergy > 95
        if energy == 75:
            mask = finalEnergy > 70
        if energy == 50:
            mask = finalEnergy > 45
        if energy == 25:
            mask = finalEnergy > 20
        else:
            mask = np.ones_like(finalEnergy, dtype=bool)
        finalDirectionCosineX = finalDirectionCosineX[mask]
        finalDirectionCosineY = finalDirectionCosineY[mask]
        finalEnergy = finalEnergy[mask]
        initialEnergy = initialEnergy[mask]
        isSign = isSign[mask]
        
        logFinalEnergy = np.log((initialEnergy - finalEnergy) / initialEnergy) / np.sqrt(initialEnergy) 
        energyCa, doseCalculated = calculateEnergyLossDose(finalEnergy, energy, mass)
        
        # Extract values of dose and energy
        energySim, doseSim = returnDoseEnergyValue(energyFile, doseFile)
        discardedData += initialData - len(logFinalEnergy)
        
        if args.uniform:
            # Set up uniform thresholds if selected
            # threshold = [0, -0.6, 300] [9.999999999999999e-06, 9.999999999999999e-06, 3.99999960000004e-05, 2.99999970000003e-05, 3.99999960000004e-05, 2.00000040000008e-05, 2.0000010000005e-05, 8.000016000032e-05, 3.0000882025931565e-05, 0.00024002810729136382, 0.3848261497453506]
            # threshold = [0, -0.57, 300] [9.999999999999999e-06, 9.999999999999999e-06, 3.99999960000004e-05, 2.99999970000003e-05, 3.99999960000004e-05, 2.00000040000008e-05, 0.0004700002350001175, 8.000016000032e-05, 3.0000882025931565e-05, 0.00024002810729136382, 0.3848261497453506]
            # threshold = [0, -0.57, 200] # [7e-05, 7.999999999999999e-05, 0.0001399999860000014, 0.0001399999860000014, 0.0001699999830000017, 0.00019000003800000758, 0.0007700003850001926, 0.00047000094000188, 0.0006000176405186312, 0.0019602295428794714, 0.3935531985500457]
            # threshold = [0, -0.57, 150] [0.00031, 0.00033, 0.0004199999580000042, 0.0004499999550000045, 0.0005599999440000056, 0.00075000015000003, 0.001660000830000415, 0.00239000478000956, 0.003020088790610444, 0.007570886550815101, 0.4065316251018878]
            # threshold = [0, -0.57, 100] [0.0014399999999999999, 0.00156, 0.0018399998160000186, 0.0021999997800000223, 0.002819999718000028, 0.0035800007160001427, 0.00536000268000134, 0.00978001956003912, 0.016050471883873384, 0.03782442924066408, 0.4498814049121254]
            # threshold = [0, -0.57, 70] # [0.0038399999999999997, 0.00424, 0.0048699995130000485, 0.005649999435000057, 0.007179999282000071, 0.00951000190200038, 0.014140007070003534, 0.026280052560105124, 0.04847142505989676, 0.11785380068005963, 0.5577664872150264]
            # [0.009870974265159971, 0.011141241134262358, 0.01253157020574678, 0.014512105706538018, 0.017823177872614688, 0.02293526134895345, 0.03481215292258528, 0.07033017354580758, 0.1780700528429444, 0.42076627625166035]
            threshold = [0, -0.6, 70] #  [0.005770332948211112, 0.006300396925006276, 0.007350538794493637, 0.0084407107078416, 0.010611123718001736, 0.013831915720327265, 0.02062416814438198, 0.03858494780878112, 0.07267490589140096, 0.18311610589112648, 0.6593214482150853]
            # [0.014962238350857289, 0.016832832965788143, 0.019053629716460987, 0.02203485427839753, 0.026827197737152874, 0.03544256084356296, 0.053138274876061556, 0.10872923247633355, 0.28250712360431307, 0.914612187572127]
            # threshold = [0, -0.57, 50] # [0.00896080288793876, 0.009970994108112578, 0.011441306597213402, 0.01344180389008205, 0.01638268020648178, 0.021664696906289284, 0.03167990313772085, 0.0613276919995029, 0.11691996488996978, 0.2978392211883536, 0.8708513593202599]
            # [0.023865694354673027, 0.026927248815381104, 0.03050930533812813, 0.03534248650048062, 0.04322868343698147, 0.057182685623102164, 0.08529278032945513, 0.17959386161064247, 0.4737597831209324, 1.4771631001820411]
            uniformMaxEnergy = threshold[0]
            uniformMinEnergy = threshold[1]
            uniformAngleThreshold = threshold[2]
            
            mask = (logFinalEnergy < uniformMaxEnergy) & (logFinalEnergy > uniformMinEnergy)
            filteredFinalDirectionCosineX = finalDirectionCosineX[mask]
            filteredFinalDirectionCosineY = finalDirectionCosineY[mask]
            filteredIsSign = isSign[mask]
            logFinalEnergy = logFinalEnergy[mask]

            discardedData += len(finalEnergy) - len(logFinalEnergy)
        else:
            # No filtering
            filteredFinalDirectionCosineX = finalDirectionCosineX
            filteredFinalDirectionCosineY = finalDirectionCosineY
            filteredIsSign = isSign

        # Compute final angles
        finalAngles = []
        indexToDelete = []

        for j, (directionX, directionY, sign) in enumerate(zip(filteredFinalDirectionCosineX, filteredFinalDirectionCosineY, filteredIsSign)):
            value = 1 - directionX**2 - directionY**2
            value = np.maximum(value, 0)
            directionZ = np.sqrt(value)

            if sign == 0:
                directionZ *= -1   
            angle = np.degrees(np.arccos(directionZ))
            angle *= np.sqrt(energy)
            
            if args.uniform:
                if angle > uniformAngleThreshold:
                    indexToDelete.append(j)
                    discardedData += 1
                else:
                    finalAngles.append(angle)
            else :
                finalAngles.append(angle)
      
        if indexToDelete:
            indexToDelete = np.array(indexToDelete, dtype=int)
            logFinalEnergy = np.delete(logFinalEnergy, indexToDelete)
            finalEnergy = np.delete(finalEnergy, indexToDelete)
        
        energyCut, doseCut = calculateEnergyLossDose(finalEnergy, energy, mass)
        percentajeDiscarded.append((discardedData / len(finalEnergy)) * 100)

        # Save results for energy and dose
        energySimulation.append(energySim)
        doseSimulation.append(doseSim)  
        energyCutOff.append(energyCut)
        doseCutOff.append(doseCut)
        doseCalculatedSim.append(doseCalculated)
        
        relativeDifferenceDose.append((doseCalculated - doseCut) / doseCut * 100)
        relativeDifferenceEnergy.append((energyCa - energyCut) / energyCut * 100)
        
        # Plot histograms
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))

        sns.histplot(logFinalEnergy, bins=numberOfBinsEnergies, edgecolor="black", color='orange', kde=False, ax=axs[0])
        axs[0].set_xlabel(r'$\frac{ln((E_i-E_f)/E_i)}{\sqrt{E_i}}$ (MeV$^{-1/2}$)')
        #axs[0].set_title('Final Energy distribution')
        axs[0].set_yscale('log')
         
        sns.histplot(finalAngles, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False, ax=axs[1])
        axs[1].set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
        #axs[1].set_title('Final Angles distribution')
        axs[1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(f'{savePath}OutputHistograms{energy}MeV_{args.uniform}.pdf')
        plt.close(fig)

        # Compute 2D Histogram
        if args.uniform:
            hist1, xedges1, yedges1 = np.histogram2d(finalAngles, logFinalEnergy, bins=jointNumberOfBins, range = ([0, uniformAngleThreshold], [uniformMinEnergy, uniformMaxEnergy]))
        else:
            hist1, xedges1, yedges1 = np.histogram2d(finalAngles, logFinalEnergy, bins=jointNumberOfBins)
        finalProbabilities = hist1 / np.sum(hist1)
        
        # Avoid log(0) by adding a small constant and then converting to dB
        log_probabilities_dB = 10 * np.log10(finalProbabilities + 1e-12)

        fig2, axs2 = plt.subplots(figsize=(10, 6.67))
        h1 = axs2.pcolormesh(xedges1, yedges1, log_probabilities_dB.T, cmap='Reds', shading='auto')
        fig2.colorbar(h1, ax=axs2, label='Probability (dB)')
        axs2.set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
        axs2.set_ylabel(r'$\frac{ln((E_i-E_f)/E_i)}{\sqrt{E_i}}$ (MeV$^{-1/2}$)')

        plt.tight_layout()
        plt.savefig(f'{savePath}Output2DHistograms{energy}MeV_{args.uniform}.pdf')
        plt.close(fig2) 

    except Exception as e:
        print(f'Error loading {fileName}: {e}')

doseCutOff = np.array(doseCutOff)
# Plot histograms
plt.figure(figsize=(8, 6))
plt.scatter(energies, doseCutOff, # linestyle = '-', 
         marker = 's', s = 2, color="red", label=f"Dose Cut Off") # marker = '.' 
plt.scatter(energies, doseCalculatedSim, # linestyle = '--', 
         marker = 's', s = 2, color="blue", label=f"Dose Simulation") # marker = '.' 
plt.xlabel("Energy (MeV)")
plt.yscale("log")
plt.ylabel("Dose (Gy)")
plt.legend()
            
plt.savefig(f'{savePath}DosePlot.pdf')
# plt.show()
plt.close()

# Save results to CSV
df = pd.DataFrame({
    'Energy (MeV)': energies,
    'Dose Cut Off (Gy)': doseCutOff,
    'Dose Simulation (Gy)': doseCalculatedSim,
    'Percentage of Discarded Data (%)': percentajeDiscarded,
    'relativeDifferenceDose (%)': relativeDifferenceDose,
    'relativeDifferenceEnergy (%)': relativeDifferenceEnergy,
})
df.to_csv(f'{savePath}ResultsWater.csv', index=False)
