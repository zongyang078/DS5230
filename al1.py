import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/Iris.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#Left plot: scatter of all sepal length vs sepal width
axes[0].scatter(df["SepalLengthCm"], df["SepalWidthCm"])
axes[0].set_xlabel("Sepal_Length(units=cm")
axes[0].set_ylabel("Sepal_Width(units=cm")
axes[0].set_title("No grouping")

#Right Plot: scatter with groupings by species
for species in df["Species"].unique():
    subset = df[df["Species"] == species]
    axes[1].scatter(
        subset["SepalLengthCm"],
        subset["SepalWidthCm"],
        label=species
    )

axes[1].set_xlabel("Sepal_Length(units=cm")
axes[1].set_ylabel("Sepal_Width(units=cm")
axes[1].set_title("Grouping")
axes[1].legend()

plt.tight_layout()
plt.show()