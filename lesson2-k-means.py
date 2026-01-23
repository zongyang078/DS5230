# Zongyang Li
# 1/22/2026
# SepalLengCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data/Iris.csv", sep = ",")
petal_L = data["PetalLengthCm"]
petal_W = data["PetalWidthCm"]
species = data["Species"]

x = [] # list to hold the data points
y = [] # target value
# x = [[], [], []...[]]
for i in range(len(petal_L)):
    x1 = petal_L[i]
    x2 = petal_W[i]
    x.append([x1, x2])
    y.append(species[i])
# print(y)

num_C = 3
num_inits = 10
num_max_iter = 100
km = KMeans(n_clusters = num_C, n_init = num_inits, max_iter = num_max_iter)
y_km = km.fit_predict(x)
print(y_km)
c_centers = km.cluster_centers_
print(c_centers)

k_clusters = {}
for i in range(num_C):
    k_clusters[str(i)] = [[], []]

for i in range(num_C):
    for j in range(len(y_km)):
        if(y_km[j] == i):
            n_x, n_y = x[j]
            #print(n_x, n_y)
            lists = k_clusters[str(i)]
            lists[0].append(n_x)
            lists[1].append(n_y)

figure, ax = plt.subplots(nrows = 1, ncols =2)
#plt.show()

groups = data.groupby("Species")
for name, group in groups:
    x = group.PetalLengthCm
    y = group.PetalWidthCm
    ax[0].scatter(x,y)

#print(k_clusters)
for i in k_clusters:
    x,y = k_clusters[i]
    ax[1].scatter(x,y)

plt.show()


# plt.scatter(petal_L, petal_W)

# cluster 1 = c_centers[0]
# cluster 2 = c_centers[1]
# cluster 3 = c_centers[2]
# plt.scatter(cluster1[0], cluster1[1], marker = "*", s = 200)
# plt.scatter(cluster2[0], cluster2[1], marker = "*", s = 200)
# plt.scatter(cluster3[0], cluster1[1], marker = "*", s = 200)

#plt.show()

