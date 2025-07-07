import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab

# Matplotlib params
params = {
    'xtick.labelsize': 16,    
    'ytick.labelsize': 16,      
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'legend.fontsize': 16
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

# Strip whitespace from column names
dataTable.columns = dataTable.columns.str.strip()

# Calculate estimate time for 10.000.000 histories and 4 materials
materials = 4
numberOfHistories = 10000000
numberHistoriesTable, timeRequiresTable = dataTable.iloc[-1]
estimateTimeTable = timeRequiresTable * numberOfHistories / numberHistoriesTable * materials

# Convert estimated times to days, hours, minutes, and seconds
daysTable, hoursTable, minutesTable, secsTable = secondsTodhms(estimateTimeTable)

# Plot the runtime data
plt.figure(figsize=(7.25, 6))
plt.plot(dataTable["Number of Histories"], dataTable["Time (seconds)"], marker=".")
plt.xlabel("Number of Histories")
plt.ylabel("Time (seconds)")

plt.tight_layout()
plt.savefig("./RunTimePlot.pdf")
plt.close()

# Print results
print(f"Estimated Time for Transform Variable: {daysTable} days, {hoursTable} hours, {minutesTable} minutes, {secsTable} seconds")