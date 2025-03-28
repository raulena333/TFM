import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pylab as pylab
import seaborn as sns

params = {
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,      
    'axes.titlesize' : 14,
    'axes.labelsize' : 14,
    'legend.fontsize': 14
}
pylab.rcParams.update(params)  # Apply changes

material = ["Water"]
density = [1.0]
dataForEnergies = []
percentajeDiscarded = []
dataForAngles = []
numberOfBinsAngles = 100
numberOfBinsEnergies = 100
jointNumberOfBins = 100

numberOfEnergies = 5
energies = [150, 125, 100, 75, 50]


# thresholdsAngles = [4., 5., 6., 7., 9.]
# thresholdsAngles = [2., 3., 4., 5., 7]
thresholdsAngles = [1.5, 2.2, 3., 4., 5.5]
thresholdsMaxEnergies = [-5., -5, -4.7, -4.4, -4.]
thresholdsMinEnergies = [-7, -6.4, -6, -5.4, -4.8]

# Load data from phsp file
for i in range(numberOfEnergies):   
    #Load File
    fileName = f"OutputVoxelEnergy{energies[i]}.phsp" 
    
    try:
        # Count discarded data (energies and angles)
        discardedData = 0
        
        # Load files
        newData = np.loadtxt(fileName)  
        print(f'{fileName} loaded successfully.')
        finalDirectionCosineX, finalDirectionCosineY, finalEnergy, isSign, initialEnergy, initialDirectionCosineZ = newData[:, [3,4,5,8,10,16]].T
        
        # Calculate log of the difference of initial and final energies
        logFinalEnergy = np.log((initialEnergy - finalEnergy) / initialEnergy)

        # Boolean mask for filtering first the energies
        mask = (logFinalEnergy < thresholdsMaxEnergies[i]) | (logFinalEnergy > thresholdsMinEnergies[i])

        filteredFinalEnergy = finalEnergy[mask]
        filteredFinalDirectionCosineX = finalDirectionCosineX[mask]
        filteredFinalDirectionCosineY = finalDirectionCosineY[mask]
        filteredIsSign = isSign[mask]
        
        logFinalEnergy = logFinalEnergy[mask]
        discardedData += len(finalEnergy) - len(filteredFinalEnergy)
        
        # Calculate initial and final Angles for tables, using isSign to determine whether its + or - 
        # For calculating the final direction along de Z-axis, we will use the Xcosine and Ycosine directions using the equaiton : ±\sqrt(1-d_x^2-d_y^2)
        # This formula comes from the folliwing web page: https://mathworld.wolfram.com/DirectionCosine.html
        # Calculate initial and final Angles for tables, using isSign to determine whether it's + or -
        finalAngles = []
        indexToDelete = []

        # Loop over each particle's direction cosines
        for j, (directionX, directionY, sign) in enumerate(zip(filteredFinalDirectionCosineX, filteredFinalDirectionCosineY, filteredIsSign)):
            directionZ = np.sqrt(1 - directionX**2 - directionY**2)
            
            # Adjust sign of directionZ based on isSign
            if sign == 0:
                directionZ *= -1
            angle = np.degrees(np.arccos(directionZ))
            
            # Keep only angles within the threshold
            if angle < thresholdsAngles[i]:
                finalAngles.append(angle)
            else:
                indexToDelete.append(j)
                discardedData += 1
                
        indexToDelete = np.array(indexToDelete)                     
        logFinalEnergy = np.delete(logFinalEnergy, indexToDelete)

        # Save variables for angles and energies for later use
        dataForEnergies.append(finalEnergy)
        dataForAngles.append(finalAngles)

        # Calculate percentage of discarded angles
        percentage = (discardedData/ len(finalEnergy)) * 100
        percentajeDiscarded.append(percentage)

        # Calculate initial direction along the Z-axis, just once
        # if i == 10:
        #     initialAngles = []
        #     for directionZ in initialDirectionCosineZ:
        #        angle = np.arccos(directionZ)
        #        initialAngles.append(np.degrees(angle))
        
        # Plot results for visualization as histograms
        # fig, axs = plt.subplots(2, 2, figsize=(10, 8.33))
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))

        # sns.histplot(initialEnergy, bins=30, color='blue', kde=False, ax=axs[0, 0], edgecolor="black",)
        # axs[0, 0].set_xlabel('Energy (MeV)')
        # axs[0, 0].set_title('Initial Energy distribution')
        # axs[0, 0].set_yscale('log')

        # sns.histplot(initialAngles, bins=30, edgecolor="black", color='green', kde=False, ax=axs[0, 1])
        # axs[0, 1].set_xlabel('Angle (deg)')
        # axs[0, 1].set_title('Initial Angles distribution')
        # axs[0, 1].set_yscale('log')

        sns.histplot(logFinalEnergy, bins=numberOfBinsEnergies, edgecolor="black", color='orange', kde=False, ax=axs[0])
        axs[0].set_xlabel(r' $ln(E_i-E_f / E_i)$ (MeV)')
        axs[0].set_title('Final Energy distribution')
        axs[0].set_yscale('log')
        
        sns.histplot(finalAngles, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False, ax=axs[1])
        axs[1].set_xlabel('Angle (deg)')
        axs[1].set_title('Final Angles distribution')
        axs[1].set_yscale('log')

        plt.tight_layout()
        # plt.show()
        savefileName = f'OuputPDFHistograms{i}.pdf'
        plt.savefig(savefileName)
        plt.close(fig) 

        # Calculate the probabilities for a given  final energy and  final angle
        hist1, xedges1, yedges1 = np.histogram2d(finalAngles, logFinalEnergy, bins=jointNumberOfBins)
        finalProbabilities = hist1 / np.sum(hist1)
            
        fig2, axs2 = plt.subplots(figsize=(8, 6))

        h1 = axs2.pcolormesh(xedges1, yedges1, finalProbabilities.T, cmap='Reds', shading='auto')
        fig2.colorbar(h1, ax=axs2, label='Probability') 
        axs2.set_xlabel('Angle (deg)')
        axs2.set_ylabel(r'$ln(E_i-E_f)/E_i$ (MeV)')
        # axs2.set_title('2D Histogram of Final Energy vs Final Angle (Probabilities)')

        plt.tight_layout()
        # plt.show()
        savefileName = f'OuputPDFHistograms2D{i}.pdf'
        plt.savefig(savefileName)
        plt.close(fig2)
            
    except Exception as e:
        print(f'Error loading {fileName}: {e}')
        
print(percentajeDiscarded)

# resultsForEachMaterial = {}
# # Create a key and a dictionary for initial Energy and material
# key = (initialEnergy.mean(), material[0], density[0]) 
# resultsForEachMaterial[key] = finalProbabilities

# header = f"InitialEnergy: {initialEnergy.mean()}, Material: {material[0]}, Density: {density[0]} g/cm^3"
# np.savez(
#     'finalProbabilityTableFoMaterial.npz',
#     header=header, 
#     probabilities=finalProbabilities 
# )
