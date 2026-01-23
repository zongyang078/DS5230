"""
Name: Zongyang Li
Date: 2026-01-14
Course: DS5230
HW: HW1 - Introduction

This program reads the Iris.csv file and prints each line.
"""

# Open the Iris dataset file
file_path = "data/Iris.csv"

# Open the file in read mode
f = open(file_path, "r")

# Read and print each line in the file
for line in f.readlines():
    print(line.strip())

# Close the file
f.close()