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
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,      
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'legend.fontsize': 14
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
energies = [200, 175, 150, 125, 100, 75, 50, 25, 15, 10, 9]

# Define argument parser
parser = argparse.ArgumentParser(description="Analyze and visualize simulation data with different threshold options.")
parser.add_argument('--mode', choices=['no', 'uniform', 'predefined'], default='predefined',
                    help="Mode for running the script:\n"
                         "'no' - No filtering applied.\n"
                         "'uniform' - Single threshold for all energies.\n"
                         "'predefined' - Different thresholds per energy (default).")

parser.add_argument("--variableAngle", choices=["none", "root", "dot"], default="none",
                    help="Select the variable change to apply for the angles.\n"
                    "'none' - No variable change applied.\n"
                    "'root' - Apply square root energy transformation.\n"
                    "'dot' - Apply dot energy product transformation.")

parser.add_argument("--variableEnergy", choices=["none", "lnroot", "root"], default="none",
                    help="Select the variable change to apply for the energies.\n"
                    "'none' - No variable change applied.\n"
                    "'lnroot' - Apply natural logarithm and square root energy transformation.\n"
                    "'root' - Apply square root energy transformation.")

args = parser.parse_args()

# Path for saving plots
if(args.mode == "no"):
    savePath = "./Plots/NoThreshold/"
elif(args.mode == "uniform"):
    savePath = "./Plots/UniformThreshold/"
else:
    savePath = "./Plots/PredefinedThreshold/"

# Create directory if it doesn't exist
if not os.path.exists(savePath):
    os.makedirs(savePath)
    
# Storage for results
dataForEnergies = []
percentajeDiscarded = []
dataForAngles = []

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
        logFinalEnergy = np.log((initialEnergy - finalEnergy) / initialEnergy)
        energyCa, doseCalculated = calculateEnergyLossDose(finalEnergy, energy, mass)
        
        # Extract values of dose and energy
        energySim, doseSim = returnDoseEnergyValue(energyFile, doseFile)
        
        if args.variableEnergy == "lnroot":
            logFinalEnergy *= 1 / np.log(np.sqrt(energy))
        elif args.variableEnergy == "root":
            logFinalEnergy *= 1 / np.sqrt(energy)

        if args.mode == "predefined":     
            # Define thresholds for energies and angles
            thresholdsAngles = [1.5, 2.2, 3., 4., 5.5]
            thresholdsMaxEnergies = [-5., -5, -4.7, -4.4, -4.]
            thresholdsMinEnergies = [-7, -6.4, -6, -5.4, -4.8] 
                                     
            # Apply filtering on energy
            mask = (logFinalEnergy < thresholdsMaxEnergies[i]) & (logFinalEnergy > thresholdsMinEnergies[i])
            filteredFinalDirectionCosineX = finalDirectionCosineX[mask]
            filteredFinalDirectionCosineY = finalDirectionCosineY[mask]
            filteredIsSign = isSign[mask]
            logFinalEnergy = logFinalEnergy[mask]
            finalEnergy = finalEnergy[mask]

            discardedData += len(finalEnergy) - len(logFinalEnergy)
        elif args.mode == "uniform":
            # Set up uniform thresholds if selected
            # threshold = [0, -0.6, 300] [9.999999999999999e-06, 9.999999999999999e-06, 3.99999960000004e-05, 2.99999970000003e-05, 3.99999960000004e-05, 2.00000040000008e-05, 2.0000010000005e-05, 8.000016000032e-05, 3.0000882025931565e-05, 0.00024002810729136382, 0.3848261497453506]
            # threshold = [0, -0.57, 300] [9.999999999999999e-06, 9.999999999999999e-06, 3.99999960000004e-05, 2.99999970000003e-05, 3.99999960000004e-05, 2.00000040000008e-05, 0.0004700002350001175, 8.000016000032e-05, 3.0000882025931565e-05, 0.00024002810729136382, 0.3848261497453506]
            threshold = [0, -0.57, 200] # [7e-05, 7.999999999999999e-05, 0.0001399999860000014, 0.0001399999860000014, 0.0001699999830000017, 0.00019000003800000758, 0.0007700003850001926, 0.00047000094000188, 0.0006000176405186312, 0.0019602295428794714, 0.3935531985500457]
            # threshold = [0, -0.57, 150] [0.00031, 0.00033, 0.0004199999580000042, 0.0004499999550000045, 0.0005599999440000056, 0.00075000015000003, 0.001660000830000415, 0.00239000478000956, 0.003020088790610444, 0.007570886550815101, 0.4065316251018878]
            # threshold = [0, -0.57, 100] [0.0014399999999999999, 0.00156, 0.0018399998160000186, 0.0021999997800000223, 0.002819999718000028, 0.0035800007160001427, 0.00536000268000134, 0.00978001956003912, 0.016050471883873384, 0.03782442924066408, 0.4498814049121254]
            # threshold = [0, -0.57, 70] [0.0038399999999999997, 0.00424, 0.0048699995130000485, 0.005649999435000057, 0.007179999282000071, 0.00951000190200038, 0.014140007070003534, 0.026280052560105124, 0.04847142505989676, 0.11785380068005963, 0.5577664872150264]
            # threshold = [0, -0.57, 50] # [0.00896, 0.00997, 0.011439998856000114, 0.013439998656000134, 0.016379998362000166, 0.021660004332000868, 0.03167001583500792, 0.06129012258024516, 0.11678343343294294, 0.29695477340396564, 0.8666183756372606]
            uniformMaxEnergy = threshold[0]
            uniformMinEnergy = threshold[1]
            uniformAngleThreshold = threshold[2]
            
            mask = (logFinalEnergy < uniformMaxEnergy) & (logFinalEnergy > uniformMinEnergy)
            filteredFinalDirectionCosineX = finalDirectionCosineX[mask]
            filteredFinalDirectionCosineY = finalDirectionCosineY[mask]
            filteredIsSign = isSign[mask]
            logFinalEnergy = logFinalEnergy[mask]
            finalEnergy = finalEnergy[mask]

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
            
            if args.variableAngle == "root":
                angle *= np.sqrt(energy)
            elif args.variableAngle == "dot":
                angle *= energy
            
            if args.mode == "uniform":
                if angle > uniformAngleThreshold:
                    indexToDelete.append(j)
                    discardedData += 1
                else:
                    finalAngles.append(angle)
            elif args.mode == "predefined":
                if angle > thresholdsAngles[i]:
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
        
        # Save results
        dataForEnergies.append(logFinalEnergy)
        dataForAngles.append(finalAngles)
        percentajeDiscarded.append((discardedData / len(finalEnergy)) * 100)
        
        # Save results for energy and dose
        energySimulation.append(energySim)
        doseSimulation.append(doseSim)
        energyCutOff.append(energyCut)
        doseCutOff.append(doseCut)
        doseCalculatedSim.append(doseCalculated)
        
        # Plot histograms
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))

        sns.histplot(logFinalEnergy, bins=numberOfBinsEnergies, edgecolor="black", color='orange', kde=False, ax=axs[0])
        if args.variableEnergy == "lnroot":
            axs[0].set_xlabel(r'$\frac{ln((E_i-E_f)/E_i)}{ln\sqrt{E_i}}$ (ln(MeV)$^{-1/2}$)')  
        elif args.variableEnergy == "root":
            axs[0].set_xlabel(r'$\frac{ln((E_i-E_f)/E_i)}{\sqrt{E_i}}$ (MeV$^{-1/2}$)')
        else:
            axs[0].set_xlabel(r'$ln((E_i-E_f)/E_i)$ (u.a.)')
        axs[0].set_title('Final Energy distribution')
        axs[0].set_yscale('log')
        # normalizeNewVariable = normalizeVariable(logFinalEnergy, 0, -0.47)
        # sns.histplot(normalizeNewVariable, bins=numberOfBinsEnergies, edgecolor="black", color='orange', kde=False, ax=axs[0])
        # axs[0].set_xlabel(r'\frac{2\Bigl(x - x_{\text{min}}(E_i)\Bigr)}{x_{\text{max}}(E_i) - x_{\text{min}}(E_i)} - 1 (u.a.)')
        # axs[0].set_yscale('log')
         
        sns.histplot(finalAngles, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False, ax=axs[1])
        if args.variableAngle == "root":
            axs[1].set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
        elif args.variableAngle == "dot":
            axs[1].set_xlabel(r'Angle$\cdot$Energy (deg$\cdot$MeV)')
        else:
            axs[1].set_xlabel('Angle (deg)')
        axs[1].set_title('Final Angles distribution')
        axs[1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(f'{savePath}OutputHistograms{energy}MeV_{args.variableAngle}_{args.variableEnergy}.pdf')
        plt.close(fig)

        # Compute 2D Histogram
        if(args.mode == "uniform"):
            hist1, xedges1, yedges1 = np.histogram2d(finalAngles, logFinalEnergy, bins=jointNumberOfBins, range = ([0, uniformAngleThreshold], [uniformMinEnergy, uniformMaxEnergy]))
        else:
            hist1, xedges1, yedges1 = np.histogram2d(finalAngles, logFinalEnergy, bins=jointNumberOfBins)
        finalProbabilities = hist1 / np.sum(hist1)

        fig2, axs2 = plt.subplots(figsize=(8, 6))
        # h1 = axs2.pcolormesh(xedges1, yedges1, finalProbabilities.T, cmap='Reds', shading='auto', norm=colors.LogNorm())
        h1 = axs2.pcolormesh(xedges1, yedges1, finalProbabilities.T, cmap='Reds', shading='auto')
        fig2.colorbar(h1, ax=axs2, label='Probability')
        if args.variableAngle == "root":
            axs2.set_xlabel(r'Angle$\sqrt{E_i}$ (deg$\cdot$MeV$^{1/2}$)')
        elif args.variableAngle == "dot":
            axs2.set_xlabel(r'Angle$\cdot$E_i$ (deg$\cdot$MeV)')
        else:
            axs2.set_xlabel('Angle (deg)')
            
        if args.variableEnergy == "lnroot":
            axs2.set_ylabel(r'$ln((E_i-E_f)/E_i)/ln\sqrt{E_i}$ (ln(MeV)$^{-1}$)')
        elif args.variableEnergy == "root":
            axs2.set_ylabel(r'$ln((E_i-E_f)/E_i)\sqrt{E_i}$ (MeV$^{-1/2}$)')
        else:
            axs2.set_ylabel(r'$ln((E_i-E_f)/E_i)$ (u.a.)')

        plt.tight_layout()
        plt.savefig(f'{savePath}Output2DHistograms{energy}MeV_{args.variableAngle}_{args.variableEnergy}.pdf')
        plt.close(fig2) 

    except Exception as e:
        print(f'Error loading {fileName}: {e}')

doseCutOff = np.array(doseCutOff)
# Plot histograms
plt.figure(figsize=(10, 8))
plt.scatter(energies, doseCutOff, # linestyle = '-', 
         marker = 's', s = 3, color="red", label=f"Dose Cut Off") # marker = '.' 
plt.scatter(energies, doseCalculatedSim, # linestyle = '--', 
         marker = 's', s = 3, color="blue", label=f"Dose Simulation") # marker = '.' 
plt.xlabel("Energy (MeV)")
plt.yscale("log")
plt.ylabel("Dose (Gy)")
plt.legend()
            
plt.savefig(f'{savePath}DosePlot.pdf')
# plt.show()
plt.close()

print("Percentage of discarded data per energy level:", percentajeDiscarded)
print("Dose Simulation :", doseSimulation)
print("Dose Simulation calculated :", doseCalculatedSim)
print("Dose Cut Off :", doseCutOff) 