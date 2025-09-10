import numpy as np
# 1. Load the original .npz file
loaded_data = np.load('./DenoisingDataTransSheet.npz')

# 2. Extract the specific array you want to modify
# Replace 'data' with the actual key for your 5D array.
original_array = loaded_data['histograms']

# 3. Slice the array to get the new shape
# This will change the second dimension from 51 to 20.
# The `...` is a convenient way to select all other dimensions.
new_array = original_array[:, :20, ...]

# 4. Create a new dictionary to save
# This includes the modified array and all other original arrays.
# You need to manually add all other keys from the original file.
new_data_to_save = {}
for key in loaded_data.keys():
    if key == 'histograms':  # Check if this is the array you changed
        new_data_to_save[key] = new_array
    else:
        new_data_to_save[key] = loaded_data[key]

# 5. Save the new dictionary to a new .npz file
np.savez('./DenoisingDataTransSheetSliced.npz', **new_data_to_save)

print("The new file 'DenoisingDataTransSheetSliced.npz' has been saved with the modified array.")