import numpy as np
import matplotlib.pyplot as plt

# Initial direction vector (not necessarily along Z)
initial_velocity = np.array([[0.3, 0.4, 0.866]])  # normalized vector (|v| ≈ 1)

# Initial position (in mm)
initial_position = np.array([[10.0, 5.0, 0.0]])

# Extract vz (Z-component)
vz = initial_velocity[:, 2:3]  # shape (N, 1)

# Calculate step so that dz = 1 mm
step = initial_velocity / vz

# Update position
new_position = initial_position + step

# Print results
print(f"Initial Position: {initial_position}")
print(f"Direction Vector: {initial_velocity}")
print(f"Step (scaled to dz=1 mm): {step}")
print(f"New Position: {new_position}")

# Verify Z moved by exactly 1 mm
dz = new_position[:, 2] - initial_position[:, 2]
print(f"Z displacement (should be 1.0 mm): {dz}")

# Show X/Y displacement
dx = new_position[:, 0] - initial_position[:, 0]
dy = new_position[:, 1] - initial_position[:, 1]
print(f"X displacement: {dx}")
print(f"Y displacement: {dy}")

# Plotting the movement
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(
    initial_position[0, 0], initial_position[0, 1], initial_position[0, 2],
    step[0, 0], step[0, 1], step[0, 2],
    length=1.0, normalize=False, color='blue', label='Direction Vector'
)
ax.scatter(*initial_position[0], color='green', label='Initial Position')
ax.scatter(*new_position[0], color='red', label='New Position (after 1 mm Z)')

# Set labels and limits
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
ax.set_title('Particle Movement by 1 mm in Z-direction')
ax.legend()
ax.set_box_aspect([1,1,1])
plt.tight_layout()
plt.savefig('particle_movement.png', dpi=300)
plt.close(fig)