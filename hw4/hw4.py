# Zongyang Li
# 2/12/2026
# Part 1: K-Means, DBSCAN, Mean Shift on Spotify-YouTube Dataset
# Part 2: 3D Density Map on Iris Dataset

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde
import math

# Create output folder
output_dir = "hw4_output"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# Part 1: 3D Clustering Analysis on Spotify-YouTube Dataset
# Features: Liveness, Energy, Loudness
# ============================================================

# --- Data Loading and Preprocessing ---
data = pd.read_csv("data/Spotify_Youtube.csv", sep=",")

# Extract the three columns
liveness = data["Liveness"].values
energy = data["Energy"].values
loudness = data["Loudness"].values

# Combine into feature matrix
x_raw = np.column_stack([liveness, energy, loudness])

# Standardize — important because Loudness (-45~0) has a different scale
# than Liveness (0~1) and Energy (0~1)
scaler = StandardScaler()
x = scaler.fit_transform(x_raw)

# Color list for clusters
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']


# ============================================================
# Part 1A: K-Means Clustering
# ============================================================

# --- Elbow Graph: find optimal K ---
k_range = range(1, 11)
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, max_iter=100, random_state=42)
    km.fit(x)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), inertias, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.title("K-Means Elbow Graph")
plt.xticks(list(k_range))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "kmeans_elbow.png"), dpi=150)
plt.show()

# --- Run K-Means with optimal K (choose based on elbow graph) ---
optimal_k = 3  # Adjust after viewing the elbow graph
km = KMeans(n_clusters=optimal_k, n_init=10, max_iter=100, random_state=42)
y_km = km.fit_predict(x)

# Organize clusters for plotting (use raw data for visualization)
k_clusters = {}
for i in range(optimal_k):
    k_clusters[i] = [[], [], []]

for i in range(len(y_km)):
    cluster_id = y_km[i]
    k_clusters[cluster_id][0].append(liveness[i])
    k_clusters[cluster_id][1].append(energy[i])
    k_clusters[cluster_id][2].append(loudness[i])

# --- 3D Visualization ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for i in k_clusters:
    ax.scatter(k_clusters[i][0], k_clusters[i][1], k_clusters[i][2],
               c=colors[i], label=f"Cluster {i}", alpha=0.5, s=10)

ax.set_xlabel("Liveness")
ax.set_ylabel("Energy")
ax.set_zlabel("Loudness")
ax.set_title(f"K-Means Clustering (K={optimal_k})")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "kmeans_3d.png"), dpi=150)
plt.show()


# ============================================================
# Part 1B: DBSCAN Clustering
# ============================================================

# --- K-Distance Graph: find optimal eps ---
# Rule of thumb: min_samples = 2 * dimensions = 2 * 3 = 6
min_samples_val = 6

# Compute k-nearest neighbor distances
neighbors = NearestNeighbors(n_neighbors=min_samples_val)
neighbors.fit(x)
distances, indices = neighbors.kneighbors(x)

# Sort the distances to the k-th nearest neighbor
k_distances = np.sort(distances[:, min_samples_val - 1])

plt.figure(figsize=(8, 5))
plt.plot(k_distances)
plt.xlabel("Data Points (sorted)")
plt.ylabel(f"Distance to {min_samples_val}th Nearest Neighbor")
plt.title(f"DBSCAN K-Distance Graph (k={min_samples_val})")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "dbscan_elbow.png"), dpi=150)
plt.show()

# --- Run DBSCAN with optimal eps (choose based on k-distance graph elbow) ---
optimal_eps = 0.25  # Adjust after viewing the k-distance graph
db = DBSCAN(eps=optimal_eps, min_samples=min_samples_val).fit(x)
y_db = db.labels_

# Get unique labels
unique_labels = set(y_db)
n_clusters_db = len(unique_labels) - (1 if -1 in unique_labels else 0)
n_noise = list(y_db).count(-1)
print(f"DBSCAN: {n_clusters_db} clusters, {n_noise} noise points")

# Organize clusters for plotting
db_clusters = {}
for label in unique_labels:
    db_clusters[label] = [[], [], []]

for i in range(len(y_db)):
    label = y_db[i]
    db_clusters[label][0].append(liveness[i])
    db_clusters[label][1].append(energy[i])
    db_clusters[label][2].append(loudness[i])

# --- 3D Visualization ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for label in db_clusters:
    if label == -1:
        # Noise points in gray
        ax.scatter(db_clusters[label][0], db_clusters[label][1], db_clusters[label][2],
                   c='gray', label="Noise", alpha=0.3, s=8, marker='x')
    else:
        ax.scatter(db_clusters[label][0], db_clusters[label][1], db_clusters[label][2],
                   c=colors[label % len(colors)], label=f"Cluster {label}", alpha=0.5, s=10)

ax.set_xlabel("Liveness")
ax.set_ylabel("Energy")
ax.set_zlabel("Loudness")
ax.set_title(f"DBSCAN Clustering (eps={optimal_eps}, min_samples={min_samples_val})")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "dbscan_3d.png"), dpi=150)
plt.show()


# ============================================================
# Part 1C: Mean Shift Clustering
# ============================================================

# --- Elbow Graph: test different bandwidths ---
bandwidths = np.arange(0.3, 2.1, 0.1)
n_clusters_list = []

for bw in bandwidths:
    ms = MeanShift(bandwidth=bw)
    ms.fit(x)
    n_clusters_list.append(len(ms.cluster_centers_))

plt.figure(figsize=(8, 5))
plt.plot(bandwidths, n_clusters_list, marker='o')
plt.xlabel("Bandwidth")
plt.ylabel("Number of Clusters")
plt.title("Mean Shift: Number of Clusters vs Bandwidth")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "meanshift_elbow.png"), dpi=150)
plt.show()

# --- Run Mean Shift with estimated bandwidth ---
# estimate_bandwidth automatically finds a good bandwidth
estimated_bw = estimate_bandwidth(x, quantile=0.2)
print(f"Mean Shift estimated bandwidth: {estimated_bw:.4f}")

ms = MeanShift(bandwidth=estimated_bw)
ms.fit(x)
cluster_centers = ms.cluster_centers_
y_ms = ms.labels_
n_clusters_ms = len(cluster_centers)
print(f"Mean Shift found {n_clusters_ms} clusters")

# Assign points to nearest center (same approach as class code)
y_pred = []
for i in range(len(x)):
    distances = []
    for j in range(len(cluster_centers)):
        distance = math.dist(x[i], cluster_centers[j])
        distances.append(distance)
    min_value = min(distances)
    y_pred.append(distances.index(min_value))

# Organize clusters for plotting
ms_clusters = {}
for i in range(n_clusters_ms):
    ms_clusters[i] = [[], [], []]

for i in range(len(y_pred)):
    cluster_id = y_pred[i]
    ms_clusters[cluster_id][0].append(liveness[i])
    ms_clusters[cluster_id][1].append(energy[i])
    ms_clusters[cluster_id][2].append(loudness[i])

# Inverse transform cluster centers back to original scale for plotting
centers_original = scaler.inverse_transform(cluster_centers)

# --- 3D Visualization ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for i in ms_clusters:
    ax.scatter(ms_clusters[i][0], ms_clusters[i][1], ms_clusters[i][2],
               c=colors[i % len(colors)], label=f"Cluster {i}", alpha=0.5, s=10)

# Plot cluster centers using original scale
ax.scatter(centers_original[:, 0], centers_original[:, 1], centers_original[:, 2],
           marker="*", s=300, c='black', label="Centers")

ax.set_xlabel("Liveness")
ax.set_ylabel("Energy")
ax.set_zlabel("Loudness")
ax.set_title(f"Mean Shift Clustering ({n_clusters_ms} clusters)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "meanshift_3d.png"), dpi=150)
plt.show()


# ============================================================
# Part 2: 3D Density Map of Iris Dataset
# x = Petal Length, y = Petal Width, z = Density (Gaussian KDE)
# ============================================================

# --- Load Iris dataset ---
iris = pd.read_csv("data/Iris.csv", sep=",")

# Extract petal length and petal width
petal_length = iris["PetalLengthCm"].values
petal_width = iris["PetalWidthCm"].values

# Stack into a 2D array for KDE (2 x N)
values = np.vstack([petal_length, petal_width])

# Compute Gaussian 2D Kernel Density Estimation
kde = gaussian_kde(values)

# Create a grid of points for evaluation
x_grid = np.linspace(petal_length.min() - 0.5, petal_length.max() + 0.5, 100)
y_grid = np.linspace(petal_width.min() - 0.5, petal_width.max() + 0.5, 100)
X, Y = np.meshgrid(x_grid, y_grid)

# Evaluate KDE on the grid — Z represents the density (proxy for count of duplicate entries)
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kde(positions), X.shape)

# --- 3D Surface Plot ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with a colormap
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none', alpha=0.9)

# Add a color bar to show density values
cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('PDF', fontsize=12)

# Label axes
ax.set_xlabel('Petal Length (cm)', fontsize=12)
ax.set_ylabel('Petal Width (cm)', fontsize=12)
ax.set_zlabel('Density', fontsize=12)
ax.set_title('Surface Plot of Gaussian 2D KDE - Iris Dataset', fontsize=14)

# Adjust viewing angle for better visualization
ax.view_init(elev=25, azim=235)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "part2_density_3d.png"), dpi=150)
plt.show()

print("All figures saved to hw4_output/ folder.")
