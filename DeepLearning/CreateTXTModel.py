import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap


# Assuming you have this mapping
element_names = ['H', 'C', 'N', 'O', 'Ca', 'P', 'Na', 'Mg', 'S', 'Cl', 'K', 'Fe', 'I', 'F', 'Sb', 'Sn']

# Tissues name
tissues = [
    "AdiposeTissue", "Blood", "Brain", "Breast", "CellNucleus", "EyeLens",
    "GITract", "Heart", "Kidney", "Liver", "Lung(deflated)",
    "Lymph", "Muscle", "Ovary", "Pancreas", "Cartilage", "RedMarrow",
    "Spongiosa", "YellowMarrow", "Skin", "Spleen", "Testis", "Thyroid",
    "SkeletonCortical", "SkeletonCranium", "SkeletonFemur", "SkeletonHumerus",
    "SkeletonMandible", "SkeletonRibs(2nd,6th)", "SkeletonRibs(10th)", "SkeletonSacrum",
    "SkeletonSpongiosa", "SkeletonVertebralColumn(C4)", "SkeletonVertebralColumn(D6,L3)",
    "A150", "Acrylic", "Alderson-lung", "Alderson-muscleA", "Alderson-muscleB", "AP6",
    #"AP/L2", "AP/SF1", "B100", 
    "B110", "BR12", "Ethoxyethanol",
    "EVA-28", "FrigerioGel", 
    # "FrigerioLiquid", 
    "GlycerolTrioleate", "GoodmanLiquid", 
    # "GriffithBreast", "GriffithLung", "GriffithMuscle", 
    "M3", "Magnesium", "Mylar/Melinex",
    "Nylon-6", "ParaffinWax", "PlasterOfParis", "Polyethylene", "Polystyrene", "PTFE",
    "PVC", "RF-1", "RicePowder", "RM-1", 
    # "RM/G1", "RM/L3", 
    # "RM/SR4", 
    "RossiGel", "RossiLiquid", "RW-1", "SB5", "WittLiquid", "WT1", "Water"
]

elements = np.array([
    [11.4, 59.8, 0.7, 27.8, 0.0, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.2, 11.0, 3.3, 74.5, 0.0, 0.1, 0.1, 0.0, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
    [10.7, 14.5, 2.2, 71.2, 0.0, 0.4, 0.2, 0.0, 0.2, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.6, 33.2, 3.0, 52.7, 0.0, 0.1, 0.1, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.6, 9.0, 3.2, 74.2, 0.0, 2.6, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [9.6, 19.5, 5.7, 64.6, 0.0, 0.1, 0.1, 0.0, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.6, 11.5, 2.2, 75.1, 0.0, 0.1, 0.1, 0.0, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.3, 12.1, 3.2, 73.4, 0.0, 0.1, 0.1, 0.0, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
    [10.3, 13.2, 3.0, 72.4, 0.1, 0.2, 0.2, 0.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], 
    [10.2, 13.9, 3.0, 71.6, 0.0, 0.3, 0.2, 0.0, 0.3, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.3, 10.5, 3.1, 74.9, 0.0, 0.2, 0.2, 0.0, 0.3, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.8, 4.1, 1.1, 83.2, 0.0, 0.0, 0.3, 0.0, 0.1, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.2, 14.3, 3.4, 71.0, 0.0, 0.2, 0.1, 0.0, 0.3, 0.1, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.5, 9.3, 2.4, 76.8, 0.0, 0.2, 0.2, 0.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.6, 16.9, 2.2, 69.4, 0.0, 0.2, 0.2, 0.0, 0.1, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    [9.6, 9.9, 2.2, 74.4, 0.0, 2.2, 0.5, 0.0, 0.9, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.5, 41.4, 3.4, 43.9, 0.0, 0.1, 0.0, 0.0, 0.2, 0.2, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
    [8.5, 40.4, 2.8, 36.7, 7.4, 3.4, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
    [11.5, 64.4, 0.7, 23.1, 0.0, 0.0, 0.1, 0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.0, 20.4, 4.2, 64.5, 0.0, 0.1, 0.2, 0.0, 0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.3, 11.3, 3.2, 74.1, 0.0, 0.3, 0.1, 0.0, 0.2, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.6, 9.9, 2.0, 76.6, 0.0, 0.1, 0.2, 0.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], 
    [10.4, 11.9, 2.4, 74.5, 0.0, 0.1, 0.2, 0.0, 0.1, 0.2, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0],
    [3.4, 15.5, 4.2, 43.5, 22.5, 10.3, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    
    [5.0, 21.2, 4.0, 43.5, 17.6, 8.1, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   
    [7.0, 34.5, 2.8, 36.8, 12.9, 5.5, 0.1, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [6.0, 31.4, 3.1, 36.9, 15.2, 7.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    
    [4.6, 19.9, 4.1, 43.5, 18.7, 8.6, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [6.4, 26.3, 3.9, 43.6, 13.1, 6.0, 0.1, 0.1, 0.3, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    [5.6, 23.5, 4.0, 43.4, 15.6, 7.2, 0.1, 0.1, 0.3, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], 
    [7.4, 30.2, 3.7, 43.8, 9.8, 4.5, 0.0, 0.1, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0], 
    [8.5, 40.4, 2.8, 36.7, 7.4, 3.4, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
    [6.3, 26.1, 3.9, 43.5, 13.3, 6.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
    [7.0, 28.7, 3.8, 43.7, 11.1, 5.1, 0.0, 0.1, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0], 
    [10.1, 77.7, 3.5, 5.2, 1.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.7, 0.0, 0.0],
    [8.0, 60.0, 0.0, 32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [5.7, 74.0, 2.0, 18.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
    [8.9, 66.8, 3.1, 21.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
    [8.8, 64.4, 4.1, 20.4, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
    [8.4, 69.1, 2.4, 16.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 3.1, 0.0, 0.0],
    #[12.1, 29.3, 0.8, 57.4, 0.002, 0.2, 0.1, 0.002, 0.0, 0.1, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0],
    #[12.0, 75.5, 0.8, 11.1, 0.02, 0.01, 0.1, 0.0, 0.1, 0.4, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0],
    # [6.6, 53.7, 2.2, 3.2, 17.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.7, 0.0, 0.0],
    [3.7, 37.1, 3.2, 4.8, 26.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.9, 0.0, 0.0],
    [8.7, 69.9, 2.4, 17.9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [11.2, 53.3, 0.0, 35.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [12.3, 77.3, 0.0, 10.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
    [10.0, 12.0, 4.0, 73.3, 0.0, 0.0, 0.4, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # [10.2, 12.3, 3.5, 72.9, 0.01, 0.2, 0.1, 0.02, 0.3, 0.1, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    [11.8, 77.3, 0.0, 10.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [10.2, 12.0, 3.6, 74.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # [9.4, 61.9, 3.6, 24.5, 0.6, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01],
    # [8.0, 60.8, 4.2, 24.8, 2.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02],
    # [9.0, 60.2, 2.8, 26.6, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01],
    [11.4, 65.6, 0.0, 9.2, 0.3, 0.0, 0.0, 13.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [4.2, 62.5, 0.0, 33.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [9.8, 63.7, 12.4, 14.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [15.0, 85.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2.3, 0.0, 0.0, 55.8, 23.3, 0.0, 0.0, 0.0, 18.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [14.4, 85.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [7.7, 92.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 76.0, 0.0, 0.0],
    [4.8, 38.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 56.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [14.1, 84.1, 0.0, 0.9, 0.6, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [6.2, 44.4, 0.0, 49.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [12.2, 73.4, 0.0, 6.4, 2.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # [10.2, 9.4, 2.4, 77.4, 0.0, 0.03, 0.1, 0.0, 0.1, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    # [10.2, 12.8, 2.2, 74.1, 0.0, 0.03, 0.1, 0.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    # [10.1, 73.6, 2.2, 13.7, 0.0, 0.03, 0.01, 0.003, 0.1, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    [9.8, 15.7, 3.6, 70.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [9.8, 15.6, 3.6, 71.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [13.2, 79.4, 0.0, 3.8, 2.7, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2.6, 30.6, 1.0, 38.9, 26.8, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [4.7, 0.0, 0.0, 56.8, 0.0, 10.9, 0.0, 0.0, 0.0, 0.0, 27.6, 0.0, 0.0, 0.0, 0.0, 0.0],
    [8.1, 67.2, 2.4, 19.9, 2.3, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [11.19, 0.0, 0.0, 88.81, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
])

# Convert to percentage
elements = np.round(elements / 100.0, 5) 

# Density (g/cm^3)
density = np.array([
    0.95, 1.06, 1.04, 1.02, 1.00, 1.07, 1.03, 1.06, 1.05,
    1.06, 1.05, 1.03, 1.05, 1.05, 1.04, 1.1, 1.03, 1.18, 
    0.98, 1.09, 1.06, 1.04, 1.05, 1.92, 1.61, 1.33, 1.46, 
    1.68, 1.41, 1.52, 1.29, 1.18, 1.42, 1.33, 1.12, 1.17, 
    0.32, 1.00, 1.00, 0.91, 
    # 0.92, 0.92, 1.45, 
    1.79, 0.97, 
    0.93, 0.95, 1.12, 
    # 1.08, 
    0.92, 1.07, 
    # 1.10, 0.26, 1.12, 
    1.05, 1.74, 1.40, 1.13, 0.93, 2.32, 0.92, 1.05, 2.10, 
    1.35, 0.93, 0.84, 1.03, 
    # 1.07, 1.04, 1.03, 
    1.10, 1.11, 
    0.97, 1.87, 1.72, 1.02, 1.0
])

hounsfieldUnits = np.array([
    930, 1055, 1037, 1003, 1003, 1050, 1023, 1055, 1043,
    1053, 1044,1028, 1042, 1045, 1032, 1098, 1014, 1260, 
    958, 1075, 1054, 1032, 1040, 2376, 1903, 1499, 1683, 
    2006, 1595, 1763, 1413, 1260, 1609, 1477, 1098, 1114, 
    314, 982, 995, 875, 
    # 917, 901, 1665, 
    2203, 936, 910, 
    929, 1106, 
    # 1073, 
    896, 1056, 
    # 1068, 255, 1095, 
    1050, 1859, 
    1291, 1086, 925, 3022, 911, 983, 1869, 1717, 926, 797,  
    1041, 
    # 1062, 1031, 994, 
    1081, 1090, 986, 2313, 2144,
    996, 1000
])

# Initialize model
model = RandomForestRegressor()
model.fit(elements, hounsfieldUnits)

# Get feature importances
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[sorted_idx])
plt.xticks(range(len(importances)), np.array(element_names)[sorted_idx], rotation=45)
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Relative Importance")
plt.tight_layout()
plt.savefig("FeatureImportance.png")
plt.close()

# # Other way use Shap:
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(elements)
# force_plot = shap.force_plot(explainer.expected_value, shap_values[0], feature_names=element_names)

# # Save as HTML file
# shap.save_html("shap_force_plot.html", force_plot)

predicted = model.predict(elements)

plt.figure(figsize=(6, 6))
plt.scatter(hounsfieldUnits, predicted, c='blue', label='Predicted vs Actual')
plt.plot([min(hounsfieldUnits), max(hounsfieldUnits)],
         [min(hounsfieldUnits), max(hounsfieldUnits)], 'r--', label='Ideal Fit')
plt.xlabel("Actual HU")
plt.ylabel("Predicted HU")
plt.title("Model Predictions vs Actual HU")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ModelPredictions.png")
plt.close()

# Generate pairwise regression plots for all elements vs HU
df = pd.DataFrame(elements, columns=element_names)
df['HU'] = hounsfieldUnits

n = len(element_names)
cols = 4  # number of columns in the grid
rows = (n + cols - 1) // cols

fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
axs = axs.flatten()

for i, elem in enumerate(element_names):
    sns.regplot(x=elem, y='HU', data=df, ax=axs[i])
    axs[i].set_title(f'HU vs {elem}')

# Remove empty subplots
for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.savefig("AllElements_vs_HU.png")
plt.close()

def createAndPredictHU(
    baseComposition,
    model,
    elementNames,
    modifications=None,
    normalize=True
):
    """
    Create a new material composition by modifying a base composition,
    optionally name it, and predict HU with the model.

    Parameters:
    - baseComposition: 1D numpy array with base elemental composition (length = num elements)
    - model: trained model to predict HU
    - elementNames: list of element names in order
    - modifications: dict with element modifications (element_name: new_value)
                     and optionally 'newName': str
    - normalize: bool to normalize composition to sum to 100 after modifications and scaling

    Returns:
    - predicted HU value (float)
    - final composition numpy array after modifications
    - new material name (str) or None if not provided
    """
    comp = baseComposition.copy()

    newName = None
    if modifications:
        # Extract 'newName' if present
        if 'newName' in modifications:
            newName = modifications.pop('newName')

        # Apply elemental modifications
        for elem, val in modifications.items():
            if elem in elementNames:
                idx = elementNames.index(elem)
                comp[idx] = val
            else:
                raise ValueError(f"Element '{elem}' not found in element_names.")
    #print('New Composition: ', comp)
    if normalize:
        total = comp.sum()
        # print(f"Sum of Composition: {total}")
        if total > 0:
            comp = comp / total

    compReshaped = comp.reshape(1, -1)
    predictedHU = model.predict(compReshaped)[0]

    return predictedHU, comp, newName

# # Initial Composition 
# baseBloodCompositon = elements[1, :]
# print(f"Initial Composition: {baseBloodCompositon}")
# print('------------------------------------')

# # Example Predict blood with different modifications
# modifications = [
#     # Raise iron content substantially
#     {'Fe': 0.1, 'newName': 'BloodIronOverload'}, 
#     # Lower iron content
#     {'Fe': 0.01, 'newName': 'BloodIronScarce'},
#     # Dehydrate blood by lowering water content   
#     {'O': baseBloodCompositon[element_names.index('O')] * 0.9,
#     'H': baseBloodCompositon[element_names.index('H')] * 0.9,
#     'newName': 'BloodDehydrated'},
#     # Hydrate blood
#     {'O': baseBloodCompositon[element_names.index('O')] * 1.1,
#      'H': baseBloodCompositon[element_names.index('H')] * 1.1,
#      'newName': 'BloodHydrated'}, 
#     # More plasma protein C,N,S
#     {'C': baseBloodCompositon[element_names.index('C')] * 1.15,
#      'N': baseBloodCompositon[element_names.index('N')] * 1.15,
#      'S': baseBloodCompositon[element_names.index('S')] * 1.15,
#      'newName': 'BloodPlasmaProtein'},
#     # High electrolite content Na, K, Ca
#     {'Na': baseBloodCompositon[element_names.index('Na')] * 1.5,
#      'K': baseBloodCompositon[element_names.index('K')] * 1.5,
#      'Ca': baseBloodCompositon[element_names.index('Ca')] * 1.5, 
#      'newName': 'BloodHighElectrolite'},
#     # Higher fat content, increase C,H , decrease O
#     {'C': baseBloodCompositon[element_names.index('C')] * 1.15,
#      'H': baseBloodCompositon[element_names.index('H')] * 1.15,
#      'O': baseBloodCompositon[element_names.index('O')] * 0.9,
#      'newName': 'BloodHighFat'},
#     # Elevated phosporus content
#     {'P': baseBloodCompositon[element_names.index('P')] * 4,
#      'newName': 'BloodHighPhosporus'},
#     # Low Magnesium content
#     {'Mg': baseBloodCompositon[element_names.index('Mg')] * 0.5,
#      'newName': 'BloodLowMagnesium'},
#     # High magnesium content
#     {'Mg': baseBloodCompositon[element_names.index('Mg')] * 5,
#      'newName': 'BloodHighMagnesium'},
#     # Elevated glucose level
#     {'C': baseBloodCompositon[element_names.index('C')] * 1.4,
#      'newName': 'BloodHighGlucose'},
# ]

# resultsBlood = []
# for mod in modifications:
#     predictedHU, comp, newName = createAndPredictHU(baseBloodCompositon, model, element_names, mod, normalize=True)
#     resultsBlood.append((newName, predictedHU, comp))
#     print(f"Predicted Hounsfield Unit for {newName}: {predictedHU}")
#     print(f"Composition: {comp}, Sum of Composition: {comp.sum()}")

# # Extract predicted HU from new materials
# new_predicted_HUs = np.array([item[1] for item in resultsBlood])

# plt.figure(figsize=(6, 6))

# # Plot actual vs predicted from training data
# plt.scatter(hounsfieldUnits, predicted, c='blue', label='Predicted vs Actual (Train)')

# # Plot the ideal fit line
# plt.plot([min(hounsfieldUnits), max(hounsfieldUnits)],
#          [min(hounsfieldUnits), max(hounsfieldUnits)], 'r--', label='Ideal Fit')

# # Overlay new predicted points with different color
# plt.scatter(new_predicted_HUs, new_predicted_HUs,  # x and y are same because no actual for these
#             c='green', marker='D', label='New Materials Predictions')

# plt.xlabel("Actual HU")
# plt.ylabel("Predicted HU")
# plt.title("Model Predictions vs Actual HU and New Materials")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("ModelPredictionsWithNewMaterials.png")
# plt.close()


# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# projected = pca.fit_transform(elements)

# plt.scatter(projected[:, 0], projected[:, 1], c=hounsfieldUnits, cmap='viridis', s=50)
# plt.colorbar(label='Hounsfield Unit')
# plt.title("PCA of Elemental Compositions Colored by HU")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.tight_layout()
# plt.savefig("PCA.png")
# plt.close()

# for i, component in enumerate(pca.components_[:2]):
#     print(f"PC{i+1}:")
#     for element, weight in zip(element_names, component):
#         print(f"  {element}: {weight:.3f}")


# import umap.umap_ as umap

# reducer = umap.UMAP(n_components=2, random_state=42)
# embedding = reducer.fit_transform(elements)

# plt.scatter(embedding[:, 0], embedding[:, 1], c=hounsfieldUnits, cmap='plasma', s=50)
# plt.colorbar(label='Hounsfield Unit')
# plt.title("UMAP of Elemental Compositions Colored by HU")
# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")
# plt.tight_layout()
# plt.savefig("UMAP.png")
# plt.close()

# Define your element names (adjust if needed)
numElements = len(element_names)

# def generate_random_composition(n_samples=10000, n_components=None):
#     if n_components is None or n_components == numElements:
#         # just generate full compositions
#         compositions = np.random.rand(n_samples, numElements)
#         compositions /= compositions.sum(axis=1, keepdims=True)
#         compositions *= 100
#         return compositions
    
#     # generate only n_components random features
#     partial_comps = np.random.rand(n_samples, n_components)
#     partial_comps /= partial_comps.sum(axis=1, keepdims=True)
#     partial_comps *= 100
    
#     # zeros for remaining elements
#     zeros = np.zeros((n_samples, numElements - n_components))
    
#     # concatenate partial + zeros horizontally
#     compositions = np.hstack((partial_comps, zeros))
    
#     return compositions

def generate_random_composition_selected_elements(n_samples=10000, selected_elements=None):
    """
    Generate random compositions only for selected elements,
    others are zero.

    Parameters:
    - n_samples: int, number of samples to generate
    - selected_elements: list or array of element names to include (e.g. ['H', 'N', 'Ca'])

    Returns:
    - compositions: np.array shape (n_samples, numElements) with percentages summing to 100 only across selected elements
    """
    if selected_elements is None:
        # use all elements
        compositions = np.random.rand(n_samples, numElements)
        compositions /= compositions.sum(axis=1, keepdims=True)
        compositions *= 100
        return compositions

    # Indices of selected elements in elementNames
    selected_indices = [element_names.index(el) for el in selected_elements]

    # Initialize compositions with zeros
    compositions = np.zeros((n_samples, numElements))

    # Generate random values for selected elements only
    partial_comps = np.random.rand(n_samples, len(selected_indices))
    partial_comps /= partial_comps.sum(axis=1, keepdims=True)
    partial_comps *= 100

    # Assign generated values to correct positions in compositions
    compositions[:, selected_indices] = partial_comps

    return compositions

# Predict HU for all generated compositions
randomComps = generate_random_composition_selected_elements(n_samples=10000, selected_elements=['H', 'N', 'Ca'])
predictedHUs = model.predict(randomComps)

# Plot real vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(hounsfieldUnits, predicted, c='blue', label='Training Data')

# Overlay synthetic data (random compositions)
plt.scatter(predictedHUs, predictedHUs, c='orange', alpha=0.4, label='Synthetic Compositions')

# Ideal fit line
min_hu = min(hounsfieldUnits.min(), predictedHUs.min())
max_hu = max(hounsfieldUnits.max(), predictedHUs.max())
plt.plot([min_hu, max_hu], [min_hu, max_hu], 'r--', label='Ideal Fit')

# Labels and layout
plt.xlabel("Actual HU")
plt.ylabel("Predicted HU")
plt.title("Model Predictions vs Actual HU and Random Compositions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ModelPredictions_WithSynthetic.png")
plt.close()









columnStack = np.column_stack((elements, density, hounsfieldUnits))

# Save to file
with open("Materials.txt", "w") as f:
    f.write("TissueName\tH\tC\tN\tO\tCa\tP\tNa\tMg\tS\tCl\tK\tFe\tI\tF\tSb\tSn\tDensity(g/cm^3)\tHounsfieldUnits\n")
    for i, tissue in enumerate(tissues):
        line = f"{tissue}\t" + "\t".join(map(str, columnStack[i])) + "\n"
        f.write(line)