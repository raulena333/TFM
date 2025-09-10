import json
import matplotlib.pyplot as plt

# Load the Scalene JSON
with open("profile.json", "r") as f:
    data = json.load(f)

functions = []
cpu_times = []

# Extract function info from files
for file_path, file_data in data["files"].items():
    for func in file_data.get("functions", []):
        func_name = func["line"]
        cpu_time = func.get("n_cpu_percent_python", 0.0) + func.get("n_cpu_percent_c", 0.0)
        if cpu_time > 0.1:
            functions.append(f"{func_name} ({file_path.split('/')[-1]})")
            cpu_times.append(cpu_time)

# Sort by CPU time
sorted_data = sorted(zip(cpu_times, functions), reverse=True)
cpu_times, functions = zip(*sorted_data[:20])  # Top 20

# Plot
plt.figure(figsize=(10, 6))
plt.barh(functions, cpu_times, color='skyblue')
plt.xlabel("Total CPU Time (%)")
plt.title("Top Functions by CPU Usage (Scalene)")
plt.gca().invert_yaxis()
plt.tight_layout()

# Save as PDF
plt.savefig("scalene_cpu_profile.pdf")
plt.close()
