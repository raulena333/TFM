import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
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

dataForEnergies = []
dataForAngles = []
numberOfBins = 50
# Load data from phsp file
for i in range(10, 1, -1):
    filename = f'OutputPositionPicoBragg{i}.phsp'
    try:
        newData = np.loadtxt(filename)  
        print(f'{filename} loaded successfully.')
        finalDirectionCosineX, finalDirectionCosineY, finalEnergy, isSign, initialEnergy, initialDirectionCosineZ = newData[:, [3,4,5,8,10,16]].T
    
        # Calculate initial and final Angles for tables, using isSign to determine whether its + or - 
        # For calculating the final direction along de Z-axis, we will use the Xcosine and Ycosine directions using the equaiton : ±\sqrt(1-d_x^2-d_y^2)
        # This formula comes from the folliwing web page: https://mathworld.wolfram.com/DirectionCosine.html
        finalAngles = []
        for directionX, directionY, sign in zip(finalDirectionCosineY, finalDirectionCosineX, isSign):
            directionZ = np.sqrt(1 - directionX * directionX - directionY * directionY)
            if sign == 1:
                angle = np.arccos(directionZ)
            else:
                angle = np.arccos(- directionZ)     
            finalAngles.append(np.degrees(angle))
            
        # Save variables for angles and enrgies for later use
        dataForEnergies.append(finalEnergy) 
        dataForAngles.append(finalAngles)
        
        # Calculate initial direction along the Z-axis, just once
        if i == 10:
            initialAngles = []
            for directionZ in initialDirectionCosineZ:
               angle = np.arccos(directionZ)
               initialAngles.append(np.degrees(angle))
               
        # Calculate log of the difference of initial and final energies
        logFinalEnergy = np.log((initialEnergy - finalEnergy) / initialEnergy)
        
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

        sns.histplot(logFinalEnergy, bins=numberOfBins, edgecolor="black", color='orange', kde=False, ax=axs[0])
        axs[0].set_xlabel('Energy (MeV)')
        axs[0].set_title('Final Energy distribution')
        # axs[0].set_yscale('log')
        
        sns.histplot(finalAngles, bins=numberOfBins, edgecolor="black", color='red', kde=False, ax=axs[1])
        axs[1].set_xlabel('Angle (deg)')
        axs[1].set_title('Final Angles distribution')
        # axs[1].set_yscale('log')

        plt.tight_layout()
        # plt.show()
        savefileName = f'OuputPDFHistogramsPicoBragg{i}.pdf'
        plt.savefig(savefileName)
        plt.close(fig) 

        # Calculate the probabilities for a given  final energy and  final angle
        hist1, xedges1, yedges1 = np.histogram2d(finalAngles, logFinalEnergy, bins=numberOfBins)
        finalProbabilities = hist1 / np.sum(hist1)
        
        if i == 10:
            hist2, xedges2, yedges2 = np.histogram2d(initialAngles, initialEnergy, bins=numberOfBins)
            initialProbabilities = hist2 / np.sum(hist2)

            # Plot results for visulization as 2D histogram as heatmaps for initial angles and energies
            fig1, axs1 = plt.subplots(figsize=(8, 6))
            
            h2 = axs1.pcolormesh(xedges2, yedges2, initialProbabilities.T, cmap='Blues', shading='auto')
            fig1.colorbar(h2, ax=axs1, label='Probability')
            axs1.set_xlabel('Angle (deg)')
            axs1.set_ylabel('Energy (MeV)')
            axs1.set_title('2D Histogram of Initial Energy vs Initial Angle (Probabilities)')

            plt.tight_layout()
            # plt.show()
            savefileName = f'OuputPDFHistograms2DPicoBraggInitial.pdf'
            plt.savefig(savefileName)
            plt.close(fig1)
            
        fig2, axs2 = plt.subplots(figsize=(8, 6))

        h1 = axs2.pcolormesh(xedges1, yedges1, finalProbabilities.T, cmap='Reds', shading='auto')
        fig2.colorbar(h1, ax=axs2, label='Probability') 
        axs2.set_xlabel('Angle (deg)')
        axs2.set_ylabel('Energy (MeV)')
        axs2.set_title('2D Histogram of Final Energy vs Final Angle (Probabilities)')

        plt.tight_layout()
        #plt.show()
        savefileName = f'OuputPDFHistograms2DPicoBragg{i}.pdf'
        plt.savefig(savefileName)
        plt.close(fig2)
            
    except Exception as e:
        print(f'Error loading {filename}: {e}')






# Violin plot to compare initial and final angles
# plt.figure(figsize=(8, 6))
# sns.violinplot(data=[initialAngles, finalAngles], inner="point", palette="muted")
# plt.xticks([0, 1], ['Initial Angles', 'Final Angles'])
# plt.ylabel('Angle (degrees)')
# plt.title('Comparison of Initial and Final Angles')
# plt.show()

# Interactive 3D plot of Initial Angle, Final Angle, and Final Energy
# fig = px.scatter_3d(x=initialAngles, y=finalAngles, z=finalEnergy, 
#                     labels={'x': 'Initial Angle (degrees)', 'y': 'Final Angles', 'z': 'Final Energy'},
#                     title='3D Scatter Plot of Energy vs Angles')
# fig.show()
