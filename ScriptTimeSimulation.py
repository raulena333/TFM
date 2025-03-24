import time

# Record start time
start_time = time.time()

# Your process here (Example: a loop that takes time)
for i in range(10**6):
    pass

# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print(f"Process took {elapsed_time:.4f} seconds")
