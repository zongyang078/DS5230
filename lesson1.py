# Zongyang Li
# 1/15/2026
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from fontTools.t1Lib import std_subrs

data = pd.read_csv("data/Iris.csv", sep=",")
#print (data) prints the first 5 rows and last 5 rows
#print (data["SepalLengthCm"])

x = data["SepalLengthCm"]
y = data["SepalWidthCm"]

pearson = stats.pearsonr(x,y)
#print (pearson) # goes from -1 to 1 the closer to 1 or -1 it is the strong th
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
#plt.plot(x, intercept + slope*x, "r")
#plt.scatter(x,y)
#plt.show()

groups = data.groupby("Species")
#print(groups)
for name, group in groups:
    #print(name)
    #print(group)
    x = group.SepalLengthCm
    y = group.SepalWidthCm
    plt.scatter(x, y)

    pearson = stats.pearsonr(x, y)
    print (pearson) # goes from -1 to 1 the closer to 1 or -1 it is the strong th
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, intercept + slope*x, "r")


plt.legend(["setosa", "setosa-r", "versicolor", "versicolor-r", "virgiica", "virgiica-r"])
plt.xlabel("Sepal Length Cm")
plt.ylabel("Sepel Width Cm")
plt.title("Scatter of Iris Flowers and their sepal lengths and widths")
plt.show()
