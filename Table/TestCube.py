import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def intersect_ray_with_plane(ray_origin, ray_direction, plane_point, plane_normal):
    # Normalize the ray direction
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    # Calculate the dot product between the plane normal and the ray direction
    denominator = 0
    denominator = np.dot(plane_normal, ray_direction)
    
    # If the denominator is 0, the ray is parallel to the plane, no intersection
    if np.abs(denominator) < 1e-6:
        return None, None

    # Calculate the intersection parameter t
    intermediate = plane_point - ray_origin
    t = np.dot(intermediate, plane_normal) / denominator

    # If t >= 0, the intersection is in front of the ray's origin
    if t >= 0:
        intersection_point = ray_origin + t * ray_direction
        intersection_point = np.round(intersection_point, 3)
        
        return intersection_point, plane_normal
    else:
        return None, None  # No valid intersection, it's behind the ray

# Enable interactive mode
plt.ion()
# Define the cube size (1mm x 1mm x 1mm)
cubeSize = 1  # mm

# Define theta and phi (in degrees)
theta = np.pi / 2  # degrees (0, 180)
phi = 0 # degrees (0, 360)

# Calculate the direction vector using spherical coordinates
directionX = np.sin(theta) * np.cos(phi)
directionY = np.sin(theta) * np.sin(phi)
directionZ = np.cos(theta)

# Direction vector from spherical angles
direction_vector = np.array([directionX, directionY, directionZ])

origin = np.array([- cubeSize / 2, 0, 0])

# Create figure and 3D axis
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the cube as a collection of lines (centered at (0,0,0))
corners = np.array([
    [cubeSize / 2, cubeSize / 2, cubeSize / 2],  # 1
    [cubeSize / 2, cubeSize / 2, -cubeSize / 2],  # 2
    [cubeSize / 2, -cubeSize / 2, cubeSize / 2],  # 3
    [cubeSize / 2, -cubeSize / 2, -cubeSize / 2],  # 4
    [-cubeSize / 2, cubeSize / 2, cubeSize / 2],  # 5
    [-cubeSize / 2, cubeSize / 2, -cubeSize / 2],  # 6
    [-cubeSize / 2, -cubeSize / 2, cubeSize / 2],  # 7
    [-cubeSize / 2, -cubeSize / 2, -cubeSize / 2]   # 8
])

# Define the edges of the cube by connecting the corners
edges = [
    [0, 1], [0, 2], [0, 4], [1, 3], 
    [1, 5], [2, 3], [2, 6], [3, 7], 
    [4, 5], [4, 6], [5, 7], [6, 7]
]

# Plot the edges of the cube
# for edge in edges:
    # ax.plot3D(*zip(*corners[edge]), color="b", linewidth=2)

# Define the faces of the cube: each face is a list of 4 corner indices
faces = [
    [0, 2, 6, 4],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [1, 3, 7, 5],
    [4, 5, 7, 6], 
    [0, 2, 3, 1]
]

normals = [
    [0, 0, 1],     # +Z face
    [0, 0, -1],    # -Z face
    [1, 0, 0],     # +X face
    [-1, 0, 0],    # -X face
    [0, 1, 0],     # +Y face
    [0, -1, 0]     # -Y face
]

center = [
    [0, 0, 0.5],    # +Z face center
    [0, 0, -0.5],   # -Z face center
    [0.5, 0, 0],    # +X face center
    [-0.5, 0, 0],   # -X face center
    [0, 0.5, 0],    # +Y face center
    [0, -0.5, 0]    # -Y face center
]

# Plot the faces of the cube using lines for each face
for face in faces:
    x = [corners[face[0]][0], corners[face[1]][0], corners[face[2]][0], corners[face[3]][0], corners[face[0]][0]]
    y = [corners[face[0]][1], corners[face[1]][1], corners[face[2]][1], corners[face[3]][1], corners[face[0]][1]]
    z = [corners[face[0]][2], corners[face[1]][2], corners[face[2]][2], corners[face[3]][2], corners[face[0]][2]]
    ax.plot3D(x, y, z, color='cyan', alpha=0.5)
    
# Plot the normals of the faces of the cube
# for (normal, center) in zip(normals, center):
#     ax.quiver(center[0], center[1], center[2], normal[0], normal[1], normal[2],
#         color="r", length=0.5, arrow_length_ratio=0.1, linewidth=2)
    
# Plot the direction vector from the center of the cube
ax.quiver(origin[0], origin[1], origin[2], direction_vector[0], direction_vector[1], direction_vector[2], 
          color="r", length=0.5, arrow_length_ratio=0.1, linewidth=2)

# Loop through each face's center and normal to check for intersections
for i in range(len(normals)):
    intersection, normal = intersect_ray_with_plane(origin, direction_vector, np.array(center[i]), np.array(normals[i]))
    
    if intersection is not None:
        print(f"Intersection point: {intersection} with normal {normal}")
        break  # Exit the loop after the first intersection is found
else:
    print("No intersection found.")

# Set the limits and labels
ax.set_xlim([-cubeSize, cubeSize])
ax.set_ylim([-cubeSize, cubeSize])
ax.set_zlim([-cubeSize, cubeSize])

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')

# Add title
ax.set_title('Cube of 1 mm and Direction Vector')

# Show the plot
plt.draw()

# Keep the interactive window open for further interaction if necessary
plt.ioff()  # Turn off interactive mode
plt.show()  # Show the plot if it's not already showing