import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import csv

with open("C:/Users/Admin/Downloads/WhiteHat Python/C - 132 Project/star_with_gravity.csv", "r") as csv_file:
    reader = csv.reader(csv_file)

df = pd.read_csv("star_with_gravity.csv")

mass = df["Mass"].to_list()
radius = df["Radius"].to_list()
dist = df["Distance"].to_list()
gravity = df["Gravity"].to_list()

mass.sort()
radius.sort()
gravity.sort()
plt.plot(radius, mass)
# plt.plot(radius,gravity)

plt.title("Radius & Mass of the Star")
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()

plt.plot(mass, gravity)

plt.title("Mass vs Gravity")
plt.xlabel("Mass")
plt.ylabel("Gravity")
plt.show()

plt.scatter(radius, mass)
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()

X = df.iloc[:, [3, 4]].values

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append((kmeans.inertia_))
plt.plot(range(1, 11), wcss)
plt.title("elbow method")
plt.xlabel('Number of clusters')
plt.show()
