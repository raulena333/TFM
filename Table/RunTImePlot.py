import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab

# Matplotlib params
params = {
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,      
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'legend.fontsize': 14
}
pylab.rcParams.update(params)

def secondsTodhms(seconds):
    # Calculate days, hours, minutes, and seconds
    days = seconds // (24 * 3600)  # 1 day = 24 hours * 3600 seconds
    hours = (seconds % (24 * 3600)) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return int(days), int(hours), int(minutes), int(secs)

# Read data
dataTable = pd.read_csv("./RunTimeCreateTables.csv")
dataNormalized = pd.read_csv("./RunTimeCreateTablesNormalized.csv")

# Strip whitespace from column names
dataTable.columns = dataTable.columns.str.strip()
dataNormalized.columns = dataNormalized.columns.str.strip()

# Calculate estimate time for 10.000.000 histories and 4 materials
materials = 4
numberOfHistories = 10000000
numberHistoriesTable, timeRequiresTable = dataTable.iloc[-1]
numberHistoriesNorm, timeRequiresNorm = dataNormalized.iloc[-1]
estimateTimeTable = timeRequiresTable * numberOfHistories / numberHistoriesTable * materials
estimateTimeNorm = timeRequiresNorm * numberOfHistories / numberHistoriesNorm * materials

print(estimateTimeTable, estimateTimeNorm)
# Convert estimated times to days, hours, minutes, and seconds
daysTable, hoursTable, minutesTable, secsTable = secondsTodhms(estimateTimeTable)
daysNorm, hoursNorm, minutesNorm, secsNorm = secondsTodhms(estimateTimeNorm)

# Plot the runtime data
plt.figure(figsize=(7, 6))
plt.plot(dataTable["Number of Histories"], dataTable["Time (seconds)"], marker=".", label="Transform Variable")
plt.plot(dataNormalized["Number of Histories"], dataNormalized["Time (seconds)"], marker=".", label="Normalize Variable")
plt.xlabel("Number of Histories")
plt.ylabel("Time (seconds)")
plt.legend(
    loc='best',           
    shadow=True,          
    fancybox=True,        
    framealpha=0.9,      
)
plt.tight_layout()
plt.savefig("./RunTimePlot.pdf")
plt.close()

# Print results
print(f"Estimated Time for Transform Variable: {daysTable} days, {hoursTable} hours, {minutesTable} minutes, {secsTable} seconds")
print(f"Estimated Time for Normalize Variable: {daysNorm} days, {hoursNorm} hours, {minutesNorm} minutes, {secsNorm} seconds")