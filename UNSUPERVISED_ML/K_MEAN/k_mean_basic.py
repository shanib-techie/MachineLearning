from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

df = pd.DataFrame({
    "Income":[20,30,24,25,32,80,82,85,91,22,89,17,30,78,77,30,44,19,66,40,98,100,40,51,29]
})


kmeans = KMeans(n_clusters=2,random_state=42)

kmeans.fit(df)

print(kmeans.cluster_centers_)
print(kmeans.labels_)
centroid_x = []

for i in range(2):
    centroid_x.append(
        np.mean(df.index[kmeans.labels_ == i])
    )
plt.scatter(df.index,df["Income"],c=kmeans.labels_,s=120)
plt.scatter(
    centroid_x,
    kmeans.cluster_centers_,
    marker="x",
    s=300,
    c="black",
)

plt.ylabel("INcome")
plt.xlabel("PERson number")
plt.show()