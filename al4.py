# SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data/Iris.csv", sep=",")

petal_L = data["PetalLengthCm"].tolist()
petal_W = data["PetalWidthCm"].tolist()
species = data["Species"].tolist()

# Build x as list of [PetalLength, PetalWidth]
x = [[petal_L[i], petal_W[i]] for i in range(len(petal_L))]

# Elbow Method: k = 1..10
num_inits = 10
num_max_iter = 100

sse = []
k_values = list(range(1, 11))

for k in k_values:
    km = KMeans(
        n_clusters=k,
        n_init=num_inits,
        max_iter=num_max_iter,
        random_state=42
    )
    km.fit(x)
    sse.append(km.inertia_)

plt.figure()
plt.plot(k_values, sse, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("SSE (Inertia)")
plt.title("Elbow Method for KMeans (k=1..10)")
plt.xticks(k_values)
plt.grid(True, alpha=0.3)
plt.show()